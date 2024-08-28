// import { useState } from "react";

const XAI = ({ img1, img2, option }) => {
  console.log("from XAI component", img1);

  return (
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
  );
};

export default XAI;
