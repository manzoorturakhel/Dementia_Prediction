import { useState } from "react";

import Header from "./components/Header";

import mriImage from "./assets/mri_image.jpg";
// import viteLogo from "/vite.svg";

import "./index.css";
import Form from "./components/Form";
import XAI from "./components/XAI";
import Footer from "./components/Footer";

function App() {
  const [img1, setImg1] = useState();
  const [img2, setImg2] = useState();
  const [option, setOption] = useState("");

  const getData = (prediction, option) => {
    setImg1(prediction.image1);
    setImg2(prediction.image2);
    setOption(option);
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
          <Form sendData={getData} />
          <XAI img1={img1} img2={img2} option={option} />
        </div>
      </div>
      <Footer />
    </>
  );
}

export default App;
