"use client";
import React, { useState, useRef } from 'react';
import { Upload } from 'lucide-react';
import { motion } from 'framer-motion';
import Image from 'next/image';


export function UploadFile() {
    const [isDragging, setIsDragging] = useState(false);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [fileSelect, setFileSelect] = useState<File | null>(null);
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
    const handleFileClick = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    }
    const handleFile = (file: File) => {
        if (file.type.startsWith('image/')) {
            const previewUrl = URL.createObjectURL(file);
            setPreviewUrl(URL.createObjectURL(file));
            setFileSelect(file);
        } else {
            alert('Please upload an image file.')
        }
    }

    return (
        <div className="relative bg-gradient-to-b from-black/20 min-h-screen">
            <h2 className="text-white text-5xl font-bold text-center mt-16 mb-8">Upload Image</h2>
            <div className="grid grid-cols-2 gap-8 max-w-6xl mx-auto">
                <label>
                    <input ref={fileInputRef} type="file" accept="image/*" hidden />
                    {!previewUrl
                    ? (<motion.div 
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        whileHover={{ scale: 1.02 }}
                        className={`flex flex-col border-2 border-dashed border-gray-500 rounded-lg p-8 min-h-80 items-center justify-center transition-all cursor-pointer ${
                            isDragging
                                ? 'border-yellow-400 bg-yellow-400/10'
                                : 'border-slate-700 bg-slate-800/50 hover:border-slate-600 hover:bg-slate-800'
                        }`}
                        onClick={() => fileInputRef.current?.click()}>
                        <div className="bg-slate-700 w-24 h-24 rounded-full p-6 mb-6">
                            <Upload className="w-12 h-12 text-slate-400 mb-4"/>
                        </div>
                        <p className="text-white text-xl">Drag your image here or click to browse</p>
                        </motion.div>)
                    : (<div className="min-h-80 w-full">
                        <Image 
                            src={previewUrl}
                            alt='Road Image'
                            fill
                            className="rounded-lg object-cover"/>
                    </div>)}
                </label>
                <div className="border-2 border-dashed border-slate-700 bg-slate-800/50 rounded-lg min-h-80 p-10">
                    <h2 className="text-white text-2xl">Results</h2>
                </div>
            </div>
        </div>
    )
}