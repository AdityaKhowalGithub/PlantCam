//
//  ViewController.swift
//  PlantCam
//
//  Created by Aditya Khowal on 9/5/23.
//

import UIKit
import AVKit
import Vision
class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        let captureSession = AVCaptureSession()
        guard let captureDevice = AVCaptureDevice.default(for: .video) else {return}
                
        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else {return}
        
        captureSession.addInput(input)
        
        captureSession.startRunning()
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame
        
        
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "Video Queue"))
        captureSession.addOutput(dataOutput)

        

        
        //        let request = VNCoreMLRequest(model: <#T##VNCoreMLModel#>)
//
//        VNImageRequestHandler(cgImage: <#T##CGImage#>, options: [:]).perform(<#T##requests: [VNRequest]##[VNRequest]#>)

    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        //print("capture successful", Date())
        
        guard let pixelbuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {return}
        
        guard let model = try? VNCoreMLModel(for: MobileNetV2().model) else {return}
        
        let request = VNCoreMLRequest(model: model) { (finishedReq, err) in
            
//            print(finishedReq.results)
            
            guard let results = finishedReq.results as?
                    [VNClassificationObservation] else {return}
            guard let firstObservation = results.first else {return}
            
            print(firstObservation.identifier, firstObservation.confidence)

            
        }
        


        //
        
        try? VNImageRequestHandler(cvPixelBuffer: pixelbuffer, options: [:]).perform([request])
    }


}

