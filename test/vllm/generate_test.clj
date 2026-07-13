(ns vllm.generate-test
  (:require [clojure.test :refer [deftest is]] [vllm.generate :as generate]))

(deftest deterministic-generation-prefills-stops-and-streams
  (let [position (atom 0) streamed (atom [])
        step (fn [_ _ token]
               (swap! position inc)
               {:logits (float-array (case token
                                       1 [0 0 5 0]
                                       2 [0 0 0 5]
                                       [5 0 0 0]))})
        result (generate/generate-tokens
                nil nil [0 1]
                {:step-fn step :max-tokens 8 :temperature 0
                 :eos-token-ids #{3} :on-token #(swap! streamed conj %)})]
    (is (= [2 3] (:tokens result)))
    (is (= :stop (:finish-reason result)))
    (is (= [2 3] @streamed))
    (is (= 3 @position))))
