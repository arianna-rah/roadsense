"use client";
import React, { useState, useRef, useEffect } from 'react';
import { Upload, RefreshCcw, X } from 'lucide-react';
import { motion } from 'framer-motion';
import Image from 'next/image';


export function UploadFile() {
    const [isDragging, setIsDragging] = useState(false);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    // const [fileSelect, setFileSelect] = useState<File | null>(null);
    const [resultA, setResultA] = useState<string | null>(null);
    const [resultB, setResultB] = useState<string | null>(null);
    const [isPressed, setIsPressed] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const savedPreview = localStorage.getItem('previewUrl');
        const savedResultA = localStorage.getItem('resultA');
        const savedResultB = localStorage.getItem('resultB');
        
        if (savedPreview) setPreviewUrl(savedPreview);
        if (savedResultA) setResultA(savedResultA);
        if (savedResultB) setResultB(savedResultB);
    }, []);

    const handleClick = (e: React.MouseEvent) => {
        e.preventDefault();
        setIsPressed(true);
        setPreviewUrl(null);
        setResultA(null);
        setResultB(null);
    }

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const files = e.dataTransfer.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    }
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    }

    async function handleFile(file: File) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = reader.result as string;
            setPreviewUrl(base64String);
            localStorage.setItem('previewUrl', base64String);
        };
        reader.readAsDataURL(file);
            // setFileSelect(file);
            const formData = new FormData();
            formData.append('file', file);
            try {
                const responseB = await fetch("http://localhost:8000/yolo-predict", {
                    method: "POST",
                    body: formData
                });
                const dataB = await responseB.json();
                setResultB(dataB.result_img);
                localStorage.setItem('resultB', dataB.result_img);

                const responseA = await fetch("http://localhost:8000/predict", {
                    method: "POST",
                    body: formData
                });
                const dataA = await responseA.json();
                let resultText = "Result";

                if (dataA.predicted == "dry" && dataB.num_anomalies <= 1) {
                    resultText = `There is likely ${dataA.predicted} on the road, and the road has ${dataB.num_anomalies} damages on it. Maintain normal following distance and smooth steering, braking, and acceleration.`;
                } else if (dataA.predicted == "dry" && dataB.num_anomalies > 1) {
                    resultText = `There is likely ${dataA.predicted} on the road, and the road has ${dataB.num_anomalies} damages on it. Maintain normal following distance and smooth steering, braking, and acceleration. Stay alert for sudden hazards like debris or animals, especially at higher speeds. Slow down slightly and watch for potholes, cracks, or uneven pavement that could affect steering or tires!`;
                } else if (dataA.predicted == "wet" && dataB.num_anomalies <= 1) {
                    resultText = `The road is likely ${dataA.predicted}, and it has ${dataB.num_anomalies} damages on it. Reduce speed and increase following distance, as traction and braking efficiency are reduced. Avoid sudden movements and be cautious of hydroplaning, especially just after rain begins.`;
                } else if (dataA.predicted == "wet" && dataB.num_anomalies > 1) {
                    resultText = `The road is likely ${dataA.predicted}, and it has ${dataB.num_anomalies} damages on it. Reduce speed and increase following distance, as traction and braking efficiency are reduced. Avoid sudden movements and be cautious of hydroplaning, especially just after rain begins. Watch for potholes, cracks, or uneven pavement that could affect steering or tires!`;
                } else if (dataA.predicted == "standing water" && dataB.num_anomalies <= 1) {
                    resultText = `There is likely ${dataA.predicted} on the road, and the road has ${dataB.num_anomalies} damages on it. Slow down significantly and avoid driving through deep water if possible, as it can cause loss of control or engine damage. If unavoidable, drive steadily without braking hard and test brakes afterward.`;
                } else if (dataA.predicted == "standing water" && dataB.num_anomalies > 1) {
                    resultText = `There is likely ${dataA.predicted} on the road, and the road has ${dataB.num_anomalies} damages on it. Slow down significantly and avoid driving through deep water if possible, as it can cause loss of control or engine damage. If unavoidable, drive steadily without braking hard and test brakes afterward. Watch for potholes, cracks, or uneven pavement that could affect steering or tires!`;
                } else if (dataA.predicted == "snow" && dataB.num_anomalies <= 1) {
                    resultText = `There is likely ${dataA.predicted} on the road, and the road has ${dataB.num_anomalies} damages on it. Drive slowly, accelerate gently, and brake early to prevent skidding. Increase following distance and use lower gears when descending hills for better control.`;
                } else if (dataA.predicted == "snow" && dataB.num_anomalies > 1) {
                    resultText = `There is likely ${dataA.predicted} on the road, and the road has ${dataB.num_anomalies} damages on it. Drive slowly, accelerate gently, and brake early to prevent skidding. Increase following distance and use lower gears when descending hills for better control. Watch for potholes, cracks, or uneven pavement that could affect steering or tires!`;
                } else if (dataA.predicted == "ice" && dataB.num_anomalies <= 1) {
                    resultText = `There is likely ${dataA.predicted} on the road, and the road has ${dataB.num_anomalies} damages on it.  Assume very little traction and drive at extremely reduced speeds. Avoid sudden steering, braking, or acceleration, and increase following distance dramatically.`;
                } else if (dataA.predicted == "ice" && dataB.num_anomalies > 1) {
                    resultText = `There is likely ${dataA.predicted} on the road, and the road has ${dataB.num_anomalies} damages on it. Assume very little traction and drive at extremely reduced speeds. Avoid sudden steering, braking, or acceleration, and increase following distance dramatically. Watch for potholes, cracks, or uneven pavement that could affect steering or tires!`;
                }
                setResultA(resultText);
                localStorage.setItem('resultA', resultText);
            } catch (e) {
                console.error("Error", e)
            }
        } else {
            alert('Please upload an image file.')
        }  
    }

    return (
        <div className="relative bg-gradient-to-b from-black to-slate-500 min-h-screen">
            <h2 className="text-white text-5xl font-bold text-center mt-16 mb-8">Upload Image</h2>
            <h3 className="text-slate-500 text-3xl text-center mt-5 mb-8">Ensure the center of the lower half of the image contains unobstructed road</h3>
            <div className="grid grid-cols-2 gap-8 max-w-6xl mx-auto">
                <label>
                    <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={handleFileChange}/>
                    {!previewUrl
                    ? (<motion.div 
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        whileHover={{ scale: 1.02 }}
                        className={`flex flex-col border-2 border-dashed border-gray-500 rounded-lg p-8 min-h-100 items-center justify-center transition-all cursor-pointer ${
                            isDragging
                                ? 'border-yellow-400 bg-yellow-400/10'
                                : 'border-slate-700 bg-slate-800/50 hover:border-slate-600 hover:bg-slate-800'
                        }`}
                        onClick={() => fileInputRef.current}>
                        <div className="bg-slate-700 w-24 h-24 rounded-full p-6 mb-6">
                            <Upload className="w-12 h-12 text-slate-400 mb-4"/>
                        </div>
                        <p className="text-white text-xl font-bold">Drag your image here</p>
                        <p className="text-slate-500 text-lg">or click to browse</p>
                        </motion.div>)
                    : (<div className="relative min-h-100 w-full">
                        <div className="absolute top-2 right-2 z-20 bg-red-500 w-11 h-11 rounded-full p-2 mb-2">
                            <X className="w-7 h-7 text-white mb-4" onClick={handleClick}/>
                        </div>
                        <Image 
                            src={previewUrl}
                            alt='Road Image'
                            fill
                            className="relative rounded-lg object-cover z-0"/>
                    </div>)}
                </label>
                {!resultB
                    ? (<motion.div 
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        whileHover={{ scale: 1.02 }}
                        className={`flex flex-col border-2 border-dashed border-gray-500 rounded-lg p-8 min-h-100 items-center justify-center transition-all cursor-pointer ${
                            isDragging
                                ? 'border-yellow-400 bg-yellow-400/10'
                                : 'border-slate-700 bg-slate-800/50 hover:border-slate-600 hover:bg-slate-800'
                        }`}
                        >
                            <div className="bg-slate-700 w-24 h-24 rounded-full p-6 mb-6">
                                <RefreshCcw className="w-12 h-12 text-slate-400 mb-4"/>
                            </div>
                            <p className="text-white text-xl font-bold">Loading damage results here...</p>
                            <p className="text-slate-500 text-lg">Give it a second!</p>
                        </motion.div>)
                    : (<div className="relative min-h-100 w-full">
                        <img 
                            src={resultB}
                            alt='Road Image'
                            className="w-full h-full object-cover rounded-lg"/>
                    </div>)}
            </div>
            <div className="grid grid-cols-1 max-w-6xl mx-auto mt-10">
                {!resultA
                    ? (<motion.div 
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        whileHover={{ scale: 1.02 }}
                        className={`flex flex-col border-2 border-dashed border-gray-500 rounded-lg p-8 min-h-64 items-center justify-center transition-all cursor-pointer ${
                            isDragging
                                ? 'border-yellow-400 bg-yellow-400/10'
                                : 'border-slate-700 bg-slate-800/50 hover:border-slate-600 hover:bg-slate-800'
                        }`}
                        >
                            <div className="bg-slate-700 w-24 h-24 rounded-full p-6 mb-6">
                                <RefreshCcw className="w-12 h-12 text-slate-400 mb-4"/>
                            </div>
                            <p className="text-white text-xl font-bold">Loading condition analysis here...</p>
                            <p className="text-slate-500 text-lg">Give it a second!</p>
                        </motion.div>)
                    : (<div className="relative min-h-100 w-full">
                        <div className="border-2 border-dashed border-slate-700 bg-slate-800/50 rounded-lg min-h-64 p-15">
                            <h2 className="text-white text-3xl">{resultA}</h2>
                        </div>
                    </div>)}
            </div>
        </div>
    )
}