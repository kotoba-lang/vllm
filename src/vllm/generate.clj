(ns vllm.generate
  "Autoregressive token generation over vllm.llama/step!."
  (:require [vllm.llama :as llama])
  (:import [java.util Random]))

(defn- adjusted-logits [^floats logits history repetition-penalty]
  (let [values (double-array (map double logits))]
    (when (and repetition-penalty (not= 1.0 repetition-penalty))
      (doseq [token (distinct history)]
        (when (< -1 token (alength values))
          (let [value (aget values token)]
            (aset-double values token
                         (if (neg? value) (* value repetition-penalty)
                             (/ value repetition-penalty)))))))
    values))

(defn sample-token
  "Sample one token with temperature, top-k and nucleus top-p filtering."
  [logits history ^Random random
   {:keys [temperature top-k top-p repetition-penalty]
    :or {temperature 0.8 top-k 40 top-p 0.95 repetition-penalty 1.0}}]
  (let [values (adjusted-logits logits history repetition-penalty)
        count* (alength values)]
    (if (or (zero? temperature) (<= temperature 1.0e-8))
      (reduce (fn [best i] (if (> (aget values i) (aget values best)) i best))
              0 (range 1 count*))
      (let [ranked (sort-by (fn [i] (- (aget values i))) (range count*))
            ranked (take (min count* (max 1 (long top-k))) ranked)
            maximum (apply max (map #(aget values %) ranked))
            weighted (mapv (fn [id] [id (Math/exp (/ (- (aget values id) maximum)
                                                       temperature))]) ranked)
            total (reduce + (map second weighted))
            normalized (mapv (fn [[id weight]] [id (/ weight total)]) weighted)
            nucleus (loop [remaining normalized cumulative 0.0 kept []]
                      (let [[entry & more] remaining
                            kept (conj kept entry) cumulative (+ cumulative (second entry))]
                        (if (or (>= cumulative top-p) (empty? more)) kept
                            (recur more cumulative kept))))
            nucleus-total (reduce + (map second nucleus))
            target (* (.nextDouble random) nucleus-total)]
        (loop [remaining nucleus cumulative 0.0]
          (let [[[id probability] & more] remaining
                cumulative (+ cumulative probability)]
            (if (or (>= cumulative target) (empty? more)) id
                (recur more cumulative))))))))

(defn generate-tokens
  "Prefill prompt IDs and generate tokens. Options include `:max-tokens`,
  `:seed`, `:eos-token-ids`, sampling options, and optional `:on-token`.
  `:step-fn` is an injectable `(fn [model state token] result)` test/runtime seam."
  [model state prompt-ids
   {:keys [max-tokens seed eos-token-ids on-token step-fn] :as options
    :or {max-tokens 128 seed 0 eos-token-ids #{} step-fn llama/step!}}]
  (when-not (seq prompt-ids)
    (throw (ex-info "generation requires at least one prompt token" {})))
  (let [random (Random. (long seed))
        history (transient [])]
    (doseq [token (butlast prompt-ids)]
      (step-fn model state token)
      (conj! history token))
    (loop [input (last prompt-ids) generated [] history (persistent! (conj! history input))]
      (if (>= (count generated) max-tokens)
        {:tokens generated :finish-reason :length :state state}
        (let [{:keys [logits]} (step-fn model state input)
              token (sample-token logits history random options)
              generated (conj generated token)]
          (when on-token (on-token token))
          (if (contains? (set eos-token-ids) token)
            {:tokens generated :finish-reason :stop :state state}
            (recur token generated (conj history token))))))))
