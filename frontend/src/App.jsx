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
  // the img1 and img2 are used here because we send the data from Form component to XAI component
  // as react only support one-way binding which is from parent to child not vice-versa but we can sent them using props
  // as Form and XAI component are not directly related but both are used in APP.jsx we set the img1 and img2 data in APP.jsx 
  //  from Form.jsx and then we send it from app.jsx to XAI.jsx to make an indirect connection between them
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
