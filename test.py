import numpy as np
import tensorflow as tf
import cv2
import tflite_runtime.interpreter as tflite


interpreter = tflite.Interpreter(model_path="decrypted.tflite")
interpreter.allocate_tensors()
