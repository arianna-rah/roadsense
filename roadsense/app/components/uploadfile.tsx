"use client";
import React, { useState, useRef } from 'react';
import { Upload } from 'lucide-react';
import { motion } from 'framer-motion';
import Image from 'next/image';


export function UploadFile() {
    const [isDragging, setIsDragging] = useState(false);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [fileSelect, setFileSelect] = useState<File | null>(null);
    const [resultA, setResultA] = useState("Result");
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault;
        setIsDragging(true);
    }

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault;
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
            const previewUrl = URL.createObjectURL(file);
            setPreviewUrl(URL.createObjectURL(file));
            setFileSelect(file);
            const formData = new FormData();
            formData.append('file', file);
            try {
                const response = await fetch("http://localhost:8000/predict", {
                    method: "POST",
                    body: formData
                });
                const data = await response.json();
                console.log(data);
                setResultA(`There is likely ${data.predicted} on the road. Be careful!!`);
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
                        onClick={() => fileInputRef.current?.click()}>
                        <div className="bg-slate-700 w-24 h-24 rounded-full p-6 mb-6">
                            <Upload className="w-12 h-12 text-slate-400 mb-4"/>
                        </div>
                        <p className="text-white text-xl font-bold">Drag your image here</p>
                        <p className="text-slate-500 text-lg">or click to browse</p>
                        </motion.div>)
                    : (<div className="relative min-h-100 w-full">
                        <Image 
                            src={previewUrl}
                            alt='Road Image'
                            fill
                            className="relative rounded-lg object-cover"/>
                    </div>)}
                </label>
                <div className="border-2 border-dashed border-slate-700 bg-slate-800/50 rounded-lg min-h-100 p-10">
                    <h2 className="text-white text-2xl">{resultA}</h2>
                </div>
            </div>
            <div className="grid grid-cols-1 max-w-6xl mx-auto mt-10">
                <div className="border-2 border-dashed border-slate-700 bg-slate-800/50 rounded-lg min-h-64 p-10">
                    <h2 className="text-white text-5xl">Results</h2>
                </div>
            </div>
        </div>
    )
}