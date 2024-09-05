
import Footer from '../components/Footer'
import Header from '../components/Header'
import mriImage from '../assets/background_2.jpg'

const NotFound = () => {
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
      
        <div className='text-white absolute inset-0 bg-black bg-opacity-60 flex justify-center justify-items-center'>
            <h5>Page Not Found</h5>
        </div>
        </div>
    <Footer />
    </>
  )
}

export default NotFound