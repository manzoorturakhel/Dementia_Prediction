import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";


import App from "./App.jsx";
import Results from "./routes/Results.jsx";

import "./index.css";
import NotFound from "./routes/NotFound.jsx";


const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
  },
  
  {
    path: "/results",
    element: <Results />,
  },
  {
    path: "*",
    element: <NotFound />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
