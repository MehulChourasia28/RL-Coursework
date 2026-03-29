/*
 * C++ MCTS tree for AlphaZero Gomoku.
 *
 * Only the tree operations (selection, expansion, backup, tree-reuse) live here.
 * Neural-network inference stays in Python / PyTorch so we keep the flexibility
 * of the existing training code.
 *
 * Exposed via pybind11 as  Mehuls_agent.mcts_cpp.MCTSTree
 *
 * Virtual loss: uses a separate N_inflight counter incremented on selection and
 * decremented after backup.  This reduces the PUCT exploration bonus for
 * in-flight nodes without touching W/Q, which is correct for the negamax
 * formulation where the PUCT score is  -Q + c*P*sqrt(N)/(1+N).
 * Classic W-=1 virtual loss would increase -Q (wrong sign) in negamax.
 *
 * Build:
 *   python setup_mcts.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using np_f32  = py::array_t<float,   py::array::c_style | py::array::forcecast>;
using np_i8   = py::array_t<int8_t,  py::array::c_style | py::array::forcecast>;

// ─────────────────────────────────────────────────────────────────────────────
// Inline 5-in-a-row detection
// ─────────────────────────────────────────────────────────────────────────────

static bool check_win(const int8_t* board, int size, int row, int col, int player) {
    const int D[4][2] = {{0,1},{1,0},{1,1},{1,-1}};
    for (auto& d : D) {
        int cnt = 1;
        for (int sg : {1, -1}) {
            for (int s = 1; s < 5; ++s) {
                int r = row + sg * d[0] * s;
                int c = col + sg * d[1] * s;
                if (r < 0 || r >= size || c < 0 || c >= size) break;
                if (board[r * size + c] == static_cast<int8_t>(player)) cnt++;
                else break;
            }
        }
        if (cnt >= 5) return true;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Node
// ─────────────────────────────────────────────────────────────────────────────

struct Node {
    Node*  parent;
    int    action;      // flat action that led here from parent (-1 for root)
    float  P;           // prior probability
    int    N;           // visit count (real, after backup)
    int    N_inflight;  // virtual visits currently in-flight (not yet backed up)
    float  W;           // total accumulated value (from this node's player's POV)
    float  Q;           // mean value = W / N

    bool   expanded;
    std::unordered_map<int, std::unique_ptr<Node>> children;

    Node(Node* parent_, int action_, float prior_)
        : parent(parent_), action(action_), P(prior_),
          N(0), N_inflight(0), W(0.f), Q(0.f), expanded(false) {}

    bool is_leaf() const { return !expanded; }

    // PUCT child selection.
    // Uses N + N_inflight in the denominator so in-flight nodes get a reduced
    // exploration bonus, discouraging duplicate selection within a batch.
    std::pair<int, Node*> best_child(float c_puct) const {
        float sN   = std::sqrt(static_cast<float>(std::max(N, 1)));
        float best = -1e18f;
        int   ba   = -1;
        Node* bc   = nullptr;
        for (const auto& kv : children) {
            int   a  = kv.first;
            Node* ch = kv.second.get();
            float effective_n = static_cast<float>(ch->N + ch->N_inflight);
            float s  = -ch->Q + c_puct * ch->P * sN / (1.f + effective_n);
            if (s > best) { best = s; ba = a; bc = ch; }
        }
        return {ba, bc};
    }

    // Expand this leaf by creating children for all valid actions.
    void expand(const float* priors, const int8_t* valid, int n_actions) {
        float sum = 0.f;
        for (int a = 0; a < n_actions; ++a)
            if (valid[a]) sum += priors[a];

        float inv  = (sum > 1e-10f) ? 1.f / sum : 0.f;
        int   nv   = 0;
        for (int a = 0; a < n_actions; ++a) if (valid[a]) ++nv;

        for (int a = 0; a < n_actions; ++a) {
            if (!valid[a]) continue;
            float p = (inv > 0.f) ? priors[a] * inv
                                  : (nv > 0 ? 1.f / static_cast<float>(nv) : 0.f);
            children[a] = std::make_unique<Node>(this, a, p);
        }
        expanded = true;
        if (N == 0) N = 1;   // virtual root visit so sqrt(N) is well-defined
    }

    // Propagate value back to root, flipping sign at each edge (negamax).
    void backup(float value) {
        Node* node = this;
        float v    = value;
        while (node) {
            node->N++;
            node->W += v;
            node->Q  = node->W / static_cast<float>(node->N);
            v        = -v;
            node     = node->parent;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// MCTSTree
// ─────────────────────────────────────────────────────────────────────────────

class MCTSTree {
    int   sz_;          // board_size
    int   na_;          // board_size^2
    float c_puct_;
    float dir_alpha_;
    float dir_eps_;

    std::unique_ptr<Node> root_;
    std::mt19937 rng_;

    // Stores the traversal path for each pending (non-terminal) leaf so that
    // N_inflight can be decremented after expand_and_backup().
    // Key = node_ptr (raw leaf address cast to uint64).
    std::unordered_map<uint64_t, std::vector<Node*>> pending_paths_;

    static void apply_virtual_loss(std::vector<Node*>& path) {
        for (Node* n : path) n->N_inflight++;
    }

    static void undo_virtual_loss(std::vector<Node*>& path) {
        for (Node* n : path) {
            n->N_inflight--;
            // Clamp to zero in case of any accounting edge case.
            if (n->N_inflight < 0) n->N_inflight = 0;
        }
    }

public:
    MCTSTree(int board_size, float c_puct,
             float dir_alpha = 0.3f, float dir_eps = 0.25f)
        : sz_(board_size), na_(board_size * board_size),
          c_puct_(c_puct), dir_alpha_(dir_alpha), dir_eps_(dir_eps),
          rng_(std::random_device{}()) {}

    // ── Lifecycle ────────────────────────────────────────────────────────

    void reset() {
        root_.reset();
        pending_paths_.clear();
    }

    bool root_is_null()   const { return !root_; }
    float root_q()        const { return root_ ? root_->Q : 0.f; }

    // Move root to the child reached by *action* (tree reuse).
    void advance(int action) {
        pending_paths_.clear();   // any in-flight sims are invalid after advance
        if (root_) {
            auto it = root_->children.find(action);
            if (it != root_->children.end()) {
                std::unique_ptr<Node> nr = std::move(it->second);
                nr->parent = nullptr;
                root_ = std::move(nr);
                return;
            }
        }
        root_.reset();   // unseen action → discard tree
    }

    // ── Root initialisation ───────────────────────────────────────────────

    void init_root(np_f32 priors_arr, np_i8 valid_arr) {
        if (!root_)
            root_ = std::make_unique<Node>(nullptr, -1, 1.f);
        if (root_->is_leaf())
            root_->expand(priors_arr.data(), valid_arr.data(), na_);
    }

    // Add Dirichlet noise to root children priors.
    void add_noise() {
        if (!root_ || root_->children.empty()) return;
        std::gamma_distribution<float> gamma_dist(dir_alpha_);
        const int n = static_cast<int>(root_->children.size());
        std::vector<float> eta(n);
        float sum = 0.f;
        for (auto& x : eta) { x = gamma_dist(rng_); sum += x; }
        for (auto& x : eta) x /= (sum > 1e-10f ? sum : 1.f);
        int i = 0;
        for (auto& kv : root_->children) {
            kv.second->P = (1.f - dir_eps_) * kv.second->P + dir_eps_ * eta[i++];
        }
    }

    // ── Single simulation step ────────────────────────────────────────────

    /*
     * Run one selection from root, returning either:
     *   {"terminal": True}              – terminal node; value already backed up
     *   {"terminal": False,
     *    "board":    (sz,sz) int8,      – board at leaf for NN evaluation
     *    "player":   int,
     *    "node_ptr": uint64}            – opaque pointer for expand_and_backup()
     *
     * Virtual loss (N_inflight++) is applied to every node on the traversal
     * path and stored under node_ptr.  expand_and_backup() undoes it before
     * calling backup(), so N counts remain exact.
     */
    py::dict select_one(np_i8 board_arr, int player) {
        // ── Step 1: copy numpy data into a plain C++ vector (GIL held) ──
        const int8_t* src = board_arr.data();
        std::vector<int8_t> board(src, src + na_);

        // ── Step 2: tree traversal – pure C++, release GIL ───────────────
        bool   terminal    = false;
        Node*  leaf_node   = nullptr;
        int    leaf_player = 0;

        {
            py::gil_scoped_release release;

            Node* node = root_.get();
            int   cur  = player;
            std::vector<Node*> path;

            while (!node->is_leaf()) {
                auto [action, child] = node->best_child(c_puct_);
                if (!child) break;

                // Mark this node as in-flight before descending.
                node->N_inflight++;
                path.push_back(node);

                int r = action / sz_;
                int c = action % sz_;
                board[action] = static_cast<int8_t>(cur);

                if (check_win(board.data(), sz_, r, c, cur)) {
                    // cur just won → undo in-flight marks, then backup child.
                    undo_virtual_loss(path);
                    child->backup(-1.f);
                    terminal = true;
                    break;
                }
                node = child;
                cur  = -cur;
            }

            if (!terminal) {
                // Mark the leaf itself as in-flight.
                node->N_inflight++;
                path.push_back(node);

                // Draw: board full?
                bool full = true;
                for (auto v : board) if (v == 0) { full = false; break; }
                if (full) {
                    undo_virtual_loss(path);
                    node->backup(0.f);
                    terminal = true;
                } else {
                    leaf_node   = node;
                    leaf_player = cur;
                    // Stash path so expand_and_backup() can undo virtual loss.
                    uint64_t key = static_cast<uint64_t>(
                        reinterpret_cast<std::uintptr_t>(node));
                    pending_paths_[key] = std::move(path);
                }
            }
        }  // ── GIL re-acquired ──────────────────────────────────────────

        // ── Step 3: build Python return dict (GIL held) ──────────────────
        py::dict res;
        res["terminal"] = terminal;
        if (!terminal) {
            auto out = py::array_t<int8_t>({sz_, sz_});
            std::copy(board.begin(), board.end(), out.mutable_data());
            res["board"]    = out;
            res["player"]   = leaf_player;
            res["node_ptr"] = static_cast<uint64_t>(
                reinterpret_cast<std::uintptr_t>(leaf_node));
        }
        return res;
    }

    // ── Expand leaf and propagate value ──────────────────────────────────

    /*
     * Called after Python NN inference on the leaf returned by select_one().
     * Undoes virtual loss (N_inflight--) along the stored path, then expands
     * the node with network priors and propagates *value* to the root.
     */
    void expand_and_backup(uint64_t node_ptr,
                           np_f32   priors_arr,
                           np_i8    valid_arr,
                           float    value) {
        Node* node = reinterpret_cast<Node*>(
            static_cast<std::uintptr_t>(node_ptr));
        if (!node)
            throw std::runtime_error("expand_and_backup: null node_ptr");

        // Undo virtual loss for this leaf's traversal path.
        auto it = pending_paths_.find(node_ptr);
        if (it != pending_paths_.end()) {
            undo_virtual_loss(it->second);
            pending_paths_.erase(it);
        }

        node->expand(priors_arr.data(), valid_arr.data(), na_);
        node->backup(value);
    }

    // ── Policy extraction ─────────────────────────────────────────────────

    np_f32 get_policy(float temperature) {
        auto policy = py::array_t<float>(na_);
        float* p    = policy.mutable_data();
        std::fill(p, p + na_, 0.f);

        if (!root_) return policy;

        if (temperature == 0.f) {
            int ba = -1, bn = -1;
            for (const auto& kv : root_->children) {
                if (kv.second->N > bn) { bn = kv.second->N; ba = kv.first; }
            }
            if (ba >= 0) p[ba] = 1.f;
        } else {
            float sum = 0.f;
            for (const auto& kv : root_->children) {
                float v = std::pow(static_cast<float>(kv.second->N),
                                   1.f / temperature);
                p[kv.first] = v;
                sum += v;
            }
            if (sum > 0.f)
                for (int a = 0; a < na_; ++a) p[a] /= sum;
        }
        return policy;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// pybind11 bindings
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(mcts_cpp, m) {
    m.doc() = "C++ MCTS tree operations for AlphaZero Gomoku";

    py::class_<MCTSTree>(m, "MCTSTree")
        .def(py::init<int, float, float, float>(),
             py::arg("board_size"), py::arg("c_puct"),
             py::arg("dir_alpha") = 0.3f, py::arg("dir_eps") = 0.25f)
        .def("reset",             &MCTSTree::reset)
        .def("root_is_null",      &MCTSTree::root_is_null)
        .def("root_q",            &MCTSTree::root_q)
        .def("advance",           &MCTSTree::advance,   py::arg("action"))
        .def("init_root",         &MCTSTree::init_root,
             py::arg("priors"), py::arg("valid"))
        .def("add_noise",         &MCTSTree::add_noise)
        .def("select_one",        &MCTSTree::select_one,
             py::arg("board"), py::arg("player"))
        .def("expand_and_backup", &MCTSTree::expand_and_backup,
             py::arg("node_ptr"), py::arg("priors"),
             py::arg("valid"), py::arg("value"))
        .def("get_policy",        &MCTSTree::get_policy, py::arg("temperature"));
}
