import { useState } from "react";
import ProgressBar from "@ramonak/react-progress-bar";

import "../index.css";
import Spinner from "./Spinner";

const Form = ({ sendData }) => {
  const [classes, setClasses] = useState({
    0: "mildDemented",
    1: "ModerateDemented",
    2: "NonDemented",
    3: "VeryMildDemented",
  });
  const [filePreview, setFilePreview] = useState();
  // const [img1, setImg1] = useState();
  // const [img2, setImg2] = useState();

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
    // console.log("file", e);
    console.log("from XAI", e.target.value);
    setOptionSelected(e.target.value);
  };
  const onModelSelectHandler = (e) => {
    console.log(e.target.value);
    setModel(e.target.value);
  };
  const handleSubmit = async (event) => {
    console.log("on submit");
    event.preventDefault();
    setIsLoading((pre) => !pre);

    if (!file) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();

    formData.append("image", file);
    formData.append("model", model);
    formData.append("XAI_technique", optionSelected);

    console.log(formData, file);

    try {
      // setIsLoading((pre) => !pre);
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log("result", result);
      setPrediction(result);

      sendData(result, optionSelected);
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred during prediction.");
    }
    setIsLoading((pre) => !pre);
  };

  const PreviewImage = (e) => {
    setFilePreview(URL.createObjectURL(e.target.files[0]));
    setFile(e.target.files[0]);
    // sendData(e.target, prediction);
  };

  return (
    <div>
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
  );
};

export default Form;
