(ns vllm.scheduler
  "Bounded concurrent inference scheduler with per-model admission control."
  (:refer-clojure :exclude [run!])
  (:import [java.io Closeable]
           [java.util.concurrent ArrayBlockingQueue CompletableFuture ExecutionException
            RejectedExecutionException Semaphore ThreadPoolExecutor
            ThreadPoolExecutor$AbortPolicy TimeUnit]
           [java.util.concurrent.atomic LongAdder]))

(defrecord Scheduler [^ThreadPoolExecutor executor model-permits max-per-model
                      ^LongAdder submitted ^LongAdder completed ^LongAdder failed
                      ^LongAdder rejected]
  Closeable
  (close [_]
    (.shutdown executor)
    (when-not (.awaitTermination executor 5 TimeUnit/SECONDS)
      (.shutdownNow executor))))

(defn scheduler
  [{:keys [workers queue-capacity max-per-model]
    :or {workers (max 1 (.availableProcessors (Runtime/getRuntime)))
         queue-capacity 64 max-per-model 1}}]
  (when-not (every? pos-int? [workers queue-capacity max-per-model])
    (throw (ex-info "scheduler capacities must be positive integers" {})))
  (->Scheduler
   (ThreadPoolExecutor. workers workers 0 TimeUnit/MILLISECONDS
                        (ArrayBlockingQueue. queue-capacity)
                        (ThreadPoolExecutor$AbortPolicy.))
   (atom {}) max-per-model (LongAdder.) (LongAdder.) (LongAdder.) (LongAdder.)))

(defn- permit [scheduler model-name]
  (or (get @(:model-permits scheduler) model-name)
      (get (swap! (:model-permits scheduler)
                  #(if (contains? % model-name) %
                       (assoc % model-name (Semaphore. (:max-per-model scheduler) true))))
           model-name)))

(defn submit! [scheduler model-name thunk]
  (let [result (CompletableFuture.) p (permit scheduler model-name)
        task (reify Runnable
               (run [_]
                 (.acquire ^Semaphore p)
                 (try
                   (.complete result (thunk))
                   (.increment ^LongAdder (:completed scheduler))
                   (catch Throwable error
                     (.increment ^LongAdder (:failed scheduler))
                     (.completeExceptionally result error))
                   (finally (.release ^Semaphore p)))))]
    (try
      (.execute ^ThreadPoolExecutor (:executor scheduler) task)
      (.increment ^LongAdder (:submitted scheduler))
      result
      (catch RejectedExecutionException error
        (.increment ^LongAdder (:rejected scheduler))
        (throw (ex-info "inference queue is full" {:model model-name} error))))))

(defn run! [scheduler model-name thunk]
  (try (.get ^CompletableFuture (submit! scheduler model-name thunk))
       (catch ExecutionException error (throw (.getCause error)))))

(defn stats [scheduler]
  (let [executor ^ThreadPoolExecutor (:executor scheduler)]
    {:submitted (.sum ^LongAdder (:submitted scheduler))
     :completed (.sum ^LongAdder (:completed scheduler))
     :failed (.sum ^LongAdder (:failed scheduler))
     :rejected (.sum ^LongAdder (:rejected scheduler))
     :active (.getActiveCount executor)
     :queued (.size (.getQueue executor))
     :queue-remaining (.remainingCapacity (.getQueue executor))}))
