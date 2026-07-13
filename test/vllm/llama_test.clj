(ns vllm.llama-test
  (:require [clojure.test :refer [deftest is]] [vllm.llama :as llama]))

(defn- tensor [shape values] {:shape shape :data (float-array values)})
(defn- identity-matrix [n scale]
  (tensor [n n] (for [row (range n) column (range n)]
                  (if (= row column) scale 0.0))))

(deftest llama-step-runs-rope-gqa-swiglu-and-kv-cache
  (let [width 4 ffn 8
        tensors
        {"token_embd.weight" (tensor [5 width]
                                     [0.1 0.2 0.3 0.4, 0.4 0.3 0.2 0.1,
                                      -0.2 0.1 0.5 0.3, 0.7 -0.1 0.2 0.4,
                                      0.0 0.2 -0.3 0.6])
         "blk.0.attn_norm.weight" (tensor [width] (repeat width 1.0))
         "blk.0.attn_q.weight" (identity-matrix width 0.5)
         "blk.0.attn_k.weight" (tensor [2 width] [0.3 0.1 0 0, 0 0 0.2 0.4])
         "blk.0.attn_v.weight" (tensor [2 width] [0.2 0 0.1 0, 0 0.3 0 0.2])
         "blk.0.attn_output.weight" (identity-matrix width 0.4)
         "blk.0.ffn_norm.weight" (tensor [width] (repeat width 1.0))
         "blk.0.ffn_gate.weight" (tensor [ffn width]
                                          (take (* ffn width) (cycle [0.2 -0.1 0.3 0.1])))
         "blk.0.ffn_up.weight" (tensor [ffn width]
                                        (take (* ffn width) (cycle [0.1 0.2 -0.2 0.4])))
         "blk.0.ffn_down.weight" (tensor [width ffn]
                                          (for [r (range width) c (range ffn)]
                                            (if (= r (mod c width)) 0.1 0.0)))
         "output_norm.weight" (tensor [width] (repeat width 1.0))}
        model (llama/create-model
               {:vocab-size 5 :embedding-length width :block-count 1
                :head-count 2 :head-count-kv 1 :feed-forward-length ffn
                :context-length 4 :rope-dimension-count 2
                :rope-freq-base 10000.0 :rms-epsilon 1.0e-5}
               tensors)
        state (llama/new-state model)
        first-step (llama/step! model state 1)
        second-step (llama/step! model state 2)]
    (is (= 0 (:position first-step)))
    (is (= 1 (:position second-step)))
    (is (= 2 @(:position state)))
    (is (= 5 (alength ^floats (:logits first-step))))
    (is (every? #(Float/isFinite %) (vec (:logits second-step))))
    (is (not= (vec (:logits first-step)) (vec (:logits second-step))))
    (is (some #(not (zero? %)) (vec (first (:keys state)))))
    (is (some #(not (zero? %)) (vec (first (:values state)))))))
