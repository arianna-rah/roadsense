'use client'
import React, { useState, useEffect } from 'react';
import Image from "next/image";
import { CustomSVG } from './customsvg';

type overlayInfo = {
  src: string;
  alt: string;
  title1: string;
  title2: string;
  subtitle: string;
};

export const ScrollBlur = ({src, alt, title1, title2, subtitle}: overlayInfo) => {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // change 300 or 600 to another number to adjust fade speed
  const opacity = Math.max(1 - scrollY / 300, 0);
  const overlayOpacity = Math.min(0.5 + scrollY / 600, 0.85);
  // change 30 to another number to adjust blur speed
  const blur = Math.min(scrollY / 30, 10)
  //change 0.92 to another number to scale the title text
  const scale = Math.max(1 - scrollY / 1000, 0.92)

  return (
    <div className="relative h-screen w-full overflow-hidden">
      <div className="absolute inset-0 transition-all duration-300 ease-out"
      style={{filter: `blur(${blur}px)`, transform: `scale(${scale}px)`}}>
        <Image
          src={src}
          alt={alt}
          fill
          className="absolute-inset-0 object-cover object-[50%_75%]"
        />
      </div>
      
      <div className="absolute inset-0 bg-black opacity-50" style={{opacity: overlayOpacity}}></div>
      <div className="absolute inset-0 bg-gradient-to-b from-black/20 to indigo-500 transition-opacity duration-300"></div>

      <div className="relative z-10 flex h-full flex-col items-center justify-center p-4 transition-all duration-300"
      style={{opacity: opacity, transform: `translateY(${scrollY * 0.3}px)`}}>
        <CustomSVG className="w-24 h-24 mb-6"></CustomSVG>
        <h1 className="text-white text-7xl font-bold text-center">{title1}<span className="text-yellow-300 text-7xl font-bold text-center ml-2">{title2}</span></h1>

        <h3 className="text-white text-xl text-center mt-10">{subtitle}</h3>
      </div>
    </div>
  )
}
