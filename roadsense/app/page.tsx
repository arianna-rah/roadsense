import Image from "next/image";
import { ScrollBlur} from './components/scrollblur';
import { UploadFile } from './components/uploadfile';

function App() {
  return (
  <div>
    <ScrollBlur
      src="/road.jpg"
      alt="Road going through mountains"
      title1="Road"
      title2="Sense"
      subtitle="AI-powered road condition analysis for safer journeys" />

    <UploadFile />
  </div>
  )
}

export default App;