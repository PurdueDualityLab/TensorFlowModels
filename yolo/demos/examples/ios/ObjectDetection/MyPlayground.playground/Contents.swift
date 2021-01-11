import UIKit

import TensorFlowLite
import CoreImage

var str = "Hello, playground"
var fileURL = NSURL(string : "https://i.stack.imgur.com/R64uj.jpg")
let imageData = NSData(contentsOf: fileURL! as URL) as Data?
let image = CIImage(data: imageData!)

var filePath = Bundle.main.path(forResource: "detect", ofType: "tflite")

print(filePath)

let interp = try Interpreter(modelPath: filePath)


