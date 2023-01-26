import React from 'react'
import {Button} from '@mui/material'
import { useState } from 'react';

function FunnyButton(){
    
    return(
        <div>
            <Button variant = "contained" component="label">
                Choose Image...
                <input hidden accept="image/*" multiple type="file"/>
            </Button>
        </div>   
        
        
    )
}

export default FunnyButton