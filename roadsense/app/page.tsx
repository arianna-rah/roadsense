import Image from "next/image";
import { ScrollBlur} from './components/scrollblur';

function App() {
  return (
  <div>
    <ScrollBlur
      src="/road.jpg"
      alt="Road going through mountains"
      title1="Road"
      title2="Sense"
      subtitle="AI-powered road condition analysis for safer journeys"></ScrollBlur>

    <p>This is some text</p><br></br>
    <p>More text</p><br></br>
    <p>This is some text</p><br></br>
    <p>More text</p><br></br>
    <p>This is some text</p><br></br>
    <p>More text</p><br></br>
    <p>This is some text</p><br></br>
    <p>More text</p><br></br>
  </div>
  )
}

export default App;