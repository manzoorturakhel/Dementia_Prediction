import React from 'react'
import Header from '../components/Header'
import Footer from '../components/Footer'

// images
import mriImage from '../assets/background_2.jpg'
// Custom CNN result images
import c_loss from '../assets/c_loss.png'
import c_acc from '../assets/c_acc.png'
import c_cm from '../assets/c_cm.png'
// resnet50 results images
import r_loss from '../assets/r_loss.png'
import r_acc from '../assets/r_acc.png'
import r_cm from '../assets/r_cm.png'
// densenet121 result images
import d_loss from '../assets/d_loss.png'
import d_acc from '../assets/d_acc.png'
import d_cm from '../assets/d_cm.png'
// mobilenetv2 result images
import m_loss from '../assets/m_loss.png'
import m_acc from '../assets/m_acc.png'
import m_cm from '../assets/m_cm.png'
// efficientNet_b6 result Images
import e_loss from '../assets/e_loss.png'
import e_acc from '../assets/e_acc.png'
import e_cm from '../assets/e_cm.png'
// VGG16 result images
import v_loss from '../assets/v_loss.png'
import v_acc from '../assets/v_acc.png'
import v_cm from '../assets/v_cm.png'





const Results = () => {
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
        <div className="relative z-10 flex flex-col  ">
        <h3 className='text-white ml-2 mt-2'>Custom CNN results</h3>
        <div className=' ml-6 flex flex-col md:flex-row md:justify-evenly'>
            
            <div className='text-white'><img   src={c_loss} alt='loss values' height='70%' width='70%' /> loss values</div>
            <div className='text-white'><img   src={c_acc} alt='accuracy values' height='70%' width='70%' /> accuracy values</div>
            <div className='text-white'><img   src={c_cm} alt='confusion matrix for custom CNN' height='70%' width='70%' /> Confusion Matrix</div>

        </div>
        <hr />
        
        <h3 className=' text-white ml-2'>ResNet50 results</h3>
        <div className='ml-6 flex flex-col md:flex-row md:justify-evenly'>
            
            <div className='text-white'><img   src={r_loss} alt='loss values' height='70%' width='70%' /> loss values</div>
            <div className='text-white'><img   src={r_acc} height='70%' width='70%' /> accuracy values</div>
            <div className='text-white'><img   src={r_cm} height='70%' width='70%' /> confusion matrix</div>

        </div>
        <hr />

        <h3 className='ml-2 text-white'>DenseNet121 results</h3>

        <div className='ml-6 flex flex-col md:flex-row md:justify-evenly'>
            
            <div className='text-white'><img   src={d_loss} height='70%' width='70%' /> loss values</div>
            <div className='text-white'><img   src={d_acc} height='70%' width='70%' /> accuracy values</div>
            <div className='text-white'><img   src={d_cm} height='70%' width='70%' /> confusion matrix</div>

        </div>
        <hr />

        <h3 className='ml-2 text-white'>MobileNetV2 results</h3>

        <div className='ml-6 flex flex-col md:flex-row md:justify-evenly'>
            
            <div className='text-white'><img   src={m_loss} height='70%' width='70%' /> Loss Values</div>
            <div className='text-white'><img   src={m_acc} height='70%' width='70%' /> Accuracy Values</div>
            <div className='text-white'><img   src={m_cm} height='70%' width='70%' /> Confusion Matrix</div>

        </div>
        <hr />

        <h3 className='ml-2 text-white'>EfficentNEt_b6 results</h3>

        <div className='ml-6 flex flex-col md:flex-row md:justify-evenly'>
            
            <div className='text-white'><img   src={e_loss} height='70%' width='70%' /> loss values</div>
            <div className='text-white'><img   src={e_acc} height='70%' width='70%' /> Accuracy values</div>
            <div className='text-white'><img   src={e_cm} height='70%' width='70%' /> Confusion Matrix</div>
            
        </div>
        <hr />

        <h3 className='ml-2 text-white'>VGG16 results</h3>

        <div className='ml-6 flex flex-col md:flex-row md:justify-evenly'>
            
            <div className='text-white'><img   src={v_loss} height='70%' width='70%' /> loss values</div>
            <div className='text-white'><img   src={v_acc} height='70%' width='70%' /> Accuracy values</div>
            <div className='text-white'><img   src={v_cm} height='70%' width='70%' /> Confusion Matrix</div>
            

        </div>
        



        </div>

 
        </div>
 <Footer />
   </>
  )
}

export default Results