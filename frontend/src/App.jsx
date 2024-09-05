import { useState } from "react";



import Header from "./components/Header";

import mriImage from "./assets/mri_image.jpg";


import "./index.css";

import Footer from "./components/Footer";
import Spinner from "./components/Spinner";
import ProgressBar from "@ramonak/react-progress-bar";

function App() {
   
    const [classes, setClasses] = useState({
        0: "mildDemented",
        1: "ModerateDemented",
        2: "NonDemented",
        3: "VeryMildDemented",
      });
      const [filePreview, setFilePreview] = useState();
      const [img1, setImg1] = useState(); 
      const [img2, setImg2] = useState();
      const [option, setOption] = useState("");
    
      const [file, setFile] = useState();
      const [errorMsg, setErrorMsg] = useState("");
      const [prediction, setPrediction] = useState(null);
      const [optionSelected, setOptionSelected] = useState("GradCAM");
      const [model, setModel] = useState("customCNN");
      const [isLoading, setIsLoading] = useState(false);
    
      let Display;
    
      if (isLoading) {
        // Case 1: Loading is true (ignore whether prediction is empty or not)
        Display = <Spinner />;
      } else if (!isLoading && !prediction) {
        // Case 2: Loading is false and prediction is empty
        Display = <div></div>;
      } else if (!isLoading && prediction) {
        // Case 3: Loading is false and prediction is not empty
        Display = (
          <div>
            <h2 className="text-white">
              Prediction Result: {classes[prediction.class]}{" "}
            </h2>
            <p className="text-white">Class: {prediction.class}</p>
            <p className="text-white">
              Confidence: {(prediction.confidence * 100).toFixed(2)} %
            </p>
            <div>
              <ProgressBar
                completed={(prediction.confidence * 100).toFixed(2)}
                bgColor="#3b82f6"
              />
            </div>
          </div>
        );
      }
    
      const onOptionClickHandler = (e) => {
        setOptionSelected(e.target.value);
      };
      const onModelSelectHandler = (e) => {
        console.log(e.target.value);
        setModel(e.target.value);
      };
      const handleSubmit = async (event) => {
        // this asynchronous function sents the data from frontend to the /predict api point
        event.preventDefault();
        // whenever someone presses the upload image button the spinner will start loading until it gets the new result where it be back
        // to false and stop running 
        setIsLoading((pre) => !pre); 
    
        if (!file) {
          alert("Please select an image first!");
          return;
        }
    
        const formData = new FormData();
        //  we send three things to the /predict api point
        // the image to process and do prediction on
        // 2nd the model to choose when doing the prediction
        // 3rd the XAI technique to use for interpretability
        formData.append("image", file);
        formData.append("model", model);
        formData.append("XAI_technique", optionSelected);
    
        console.log(formData, file);
    
        try {
          // here we send the data using post method
          const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData,
          });
    
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
        //  after sending the data using the post method the post method will send us something back here we catch that result
          const result = await response.json();
          console.log("result", result);
          setPrediction(result);
    
          setImg1(result.image1)
          setImg2(result.image2)
          setOption(optionSelected)
          // this sendData is sent back to app.jsx which then sends it to XAI.jsx because XAI.jsx displays the base64 encoded images there
          // the result contains the base64 images
        } catch (error) {
          console.error("Error:", error);
          alert("An error occurred during prediction.");
        }
        setIsLoading((pre) => !pre);
      };
    
      const PreviewImage = (e) => {
        setFilePreview(URL.createObjectURL(e.target.files[0]));
        setFile(e.target.files[0]);
        
      };




  return (
    <>
      <Header />

      <div
        className="relative min-h-screen bg-cover bg-center"
        style={{
          backgroundImage: `url("${mriImage}")`,
          backgroundSize: "cover",
          backgroundRepeat: "no-repeat",
        }}
      >
        {/* <!-- Black overlay that covers the entire background --> */}
        <div className="absolute inset-0 bg-black bg-opacity-60"></div>

        {/* <!-- Content on top of the overlay --> */}
        <div className="relative z-10 flex flex-col md:flex-row md:justify-evenly">
        <div>
            {/* the form part of the app */}
      <form className=" p-6 " onSubmit={handleSubmit}>
        <label
          className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
          htmlFor="file_input"
        >
          Upload MRI file
        </label>
        <input
          className="block  text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400"
          aria-describedby="file_input_help"
          id="file_input"
          type="file"
          onChange={(e) => {
            PreviewImage(e);
          }}
        />
        <p
          className="mt-1 text-sm text-gray-500 dark:text-gray-300"
          id="file_input_help"
        >
          SVG, PNG, JPG or GIF (MAX. 800x400px).
        </p>

        {errorMsg ? (
          <p className="mt-1 text-sm text-gray-500 dark:text-red-300">
            {errorMsg}
          </p>
        ) : (
          ""
        )}
        <label
          className="block mb-2 mt-2 text-sm font-medium text-gray-900 dark:text-white"
          htmlFor="XAI_options"
        >
          Choose XAI Technique
        </label>
        <select
          id="XAI_options"
          value={optionSelected}
          onChange={onOptionClickHandler}
          className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
        >
          <option value="SmoothGradCAMpp">SmoothGradCAMpp</option>
          <option value="LIME">LIME</option>
          <option value="GradCAM">GradCAM</option>
          <option value="XGradCAM">XGradCAM</option>
        </select>

        <label
          className="block mb-2 mt-2 text-sm font-medium text-gray-900 dark:text-white"
          htmlFor="model_options"
        >
          Choose CNN Model
        </label>
        <select
          id="model_options"
          value={model}
          onChange={onModelSelectHandler}
          className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
        >
          <option value="efficient_b6">EfficientNet_b6</option>
          <option value="customCNN">Custom CNN</option>
          <option value="mobileNetV2">MobileNetV2</option>
        </select>

        <button
          className="bg-blue-500 mt-5 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          onClick={handleSubmit}
        >
          Upload Photo
        </button>
      </form>
      <br />
      <div className="">
        {filePreview && (
          <img id="preview_image" src={filePreview} alt="image description" />
        )}
      </div>
      <div className="p-6 ">{Display}</div>
    </div>
    {/* the XAI Part of the app */}
    <div className="p-6 flex flex-col justify-between">
      <div className="mt-6 mb-6 md:mr-6">
        {img1 && (
          <div>
            <p className="text-white text-sm md:text-base">
              {option === "LIME"
                ? "areas that are encouraging the top prediction"
                : "RAW CAM"}
            </p>
            <img
              id="preview_image"
              src={img1}
              alt="image description"
              className="w-full md:w-64 h-auto"
            />
          </div>
        )}
      </div>

      <div>
        {img2 && (
          <div>
            <p className="text-white text-sm md:text-base">
              {option === "LIME"
                ? "areas that contributes against the top prediction (Red Ones)"
                : option}
            </p>
            <img
              id="preview_image"
              src={img2}
              alt="image description"
              className="w-full md:w-64 h-auto"
            />
          </div>
        )}
      </div>
    </div>
        </div>
      </div>
      {/* Footer part of the App */}
     <Footer />
    </>
  );
}

export default App;
