(ns vllm.scheduler-test
  (:require [clojure.test :refer [deftest is]] [vllm.scheduler :as scheduler])
  (:import [java.util.concurrent CountDownLatch TimeUnit]))

(deftest bounded-queue-per-model-serialization-and-stats
  (with-open [s (scheduler/scheduler {:workers 2 :queue-capacity 1 :max-per-model 1})]
    (let [entered (CountDownLatch. 1) release (CountDownLatch. 1)
          first (scheduler/submit! s "m" #(do (.countDown entered)
                                               (.await release 2 TimeUnit/SECONDS)
                                               :first))]
      (is (.await entered 1 TimeUnit/SECONDS))
      (let [second (scheduler/submit! s "m" (constantly :second))
            third (scheduler/submit! s "other" (constantly :third))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"queue is full"
                              (scheduler/submit! s "overflow" (constantly :no))))
        (.countDown release)
        (is (= :first (.get first)))
        (is (= :second (.get second)))
        (is (= :third (.get third)))
        (let [stats (scheduler/stats s)]
          (is (= 3 (:submitted stats)))
          (is (= 3 (:completed stats)))
          (is (= 1 (:rejected stats))))))))
