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
    var N: Int = 20
        let identifierLabel: UILabel = {
            let label = UILabel()
            label.backgroundColor = .white
            
            label.textAlignment = .center
            label.translatesAutoresizingMaskIntoConstraints = false
            return label
        }()
        
        override func viewDidLoad() {
            super.viewDidLoad()
            
            let slider = UISlider()
            slider.minimumValue = 1
            slider.maximumValue = 100
            slider.value = Float(N)
            view.addSubview(slider)
            // Add constraints to position the slider
            slider.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                slider.centerXAnchor.constraint(equalTo: view.centerXAnchor),
                slider.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
                slider.widthAnchor.constraint(equalToConstant: 200)
            ])

            
            // Add an action to the slider
            slider.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
            
            setupIdentifierConfidenceLabel()
            
        // Do any additional setup after loading the view.
        let captureSession = AVCaptureSession()
        guard let captureDevice = AVCaptureDevice.default(for: .video) else {return}
        captureSession.sessionPreset = .photo
        guard let input = try? AVCaptureDeviceInput(device: captureDevice) else {return}
        
        captureSession.addInput(input)
        
        captureSession.startRunning()
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame
        
        let dataOutput = AVCaptureVideoDataOutput()
        dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "Video Queue"))
        captureSession.addOutput(dataOutput)
        setupIdentifierConfidenceLabel()
        
        
        
    }
    // Update the value of N when the slider's value changes
    @objc func sliderChanged(_ sender: UISlider) {
        N = Int(sender.value)
    }
    
    fileprivate func setupIdentifierConfidenceLabel() {
        view.addSubview(identifierLabel)
        identifierLabel.textColor = .black

        identifierLabel.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -32).isActive = true
        identifierLabel.leftAnchor.constraint(equalTo: view.leftAnchor).isActive = true
        identifierLabel.rightAnchor.constraint(equalTo: view.rightAnchor).isActive = true
        identifierLabel.heightAnchor.constraint(equalToConstant: 50).isActive = true
    }


    
    // Buffer to hold last N predictions
    var predictionBuffer: [String] = []
    // Add a slider to your view
   //let N = 20

   
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelbuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let config = MLModelConfiguration()
        guard let coreMLModel = try? MobileNetV2(configuration: config),
              let visionModel = try? VNCoreMLModel(for: coreMLModel.model) else {
            return
        }
        
        let request = VNCoreMLRequest(model: visionModel) { (finishedReq, err) in
            if let error = err {
                print("Error: \(error.localizedDescription)")
            } else if let results = finishedReq.results as? [VNClassificationObservation], let firstObservation = results.first {
                DispatchQueue.main.async {
                    // Add new prediction to buffer and remove oldest if buffer is full
                    self.predictionBuffer.append(firstObservation.identifier)
                    if self.predictionBuffer.count > self.N {
                        self.predictionBuffer.removeFirst()
                    }

                    // Update label with most frequent prediction in buffer
                    let mostFrequentPrediction = self.predictionBuffer.mostFrequent()
                    self.identifierLabel.text = "\(mostFrequentPrediction) \(firstObservation.confidence*100)"
                }
            } else {
                print("No results or error")
            }
        }
        
        try? VNImageRequestHandler(cvPixelBuffer: pixelbuffer, options: [:]).perform([request])
    }



}
extension Array where Element: Hashable {
    func mostFrequent() -> Element? {
        let counts = self.reduce(into: [:]) { (counts: inout [Element: Int], element) in
            counts[element, default: 0] += 1
        }
        return counts.max { $0.1 < $1.1 }?.key
    }
}
