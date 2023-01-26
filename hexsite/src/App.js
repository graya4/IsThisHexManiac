import './App.css';
import FunnyButton from './components/CustomBtn';
import { useState } from 'react';
import {Button, Box} from '@mui/material';


function App() {
  const [file, setFile] = useState();
    function handleChange(e) {
        console.log(e.target.files);
        setFile(URL.createObjectURL(e.target.files[0]));
      } 
  return (
    <div className="App">
        <div>
          <img style={{
          borderColor: 'red',
          borderWidth: 10, 
          maxWidth: '300px',
          maxHeight: '300px'
          }} src={file} />
        </div>
        <Button variant = "contained" component="label">
                Choose Image...
                <input hidden accept="image/*" multiple type="file" onChange={handleChange}/>
        </Button>
        <Button variant = "contained" component="label">
                Submit Image
                
        </Button>
    </div>
  );
}

export default App;
