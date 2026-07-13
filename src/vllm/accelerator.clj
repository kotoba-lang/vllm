(ns vllm.accelerator
  "Small execution boundary for persistent matrix accelerators.")

(defprotocol IMatrixAccelerator
  (upload-q8! [accelerator id rows columns bytes]
    "Upload one raw GGML Q8_0 matrix and return an opaque handle.")
  (gemv! [accelerator handle x]
    "Multiply a previously uploaded matrix by an F32 vector.")
  (release! [accelerator handle]
    "Release a matrix handle."))

(defprotocol IBatchedMatrixAccelerator
  (gemv-many! [accelerator requests]
    "Execute ordered `[handle f32-vector]` requests in one device submission."))

(defprotocol IQuantizedMatrixAccelerator
  (upload-quantized! [accelerator type id rows columns bytes]
    "Upload a raw GGML quantized matrix of the named type."))
