"use client";

import React from 'react';

const GalaxyBackground = () => {
  return (
    <div className="fixed inset-0 pointer-events-none select-none overflow-hidden" 
         style={{ zIndex: -1, background: '#000000' }}>
      
      {/* 🌌 Solid Deep Nebula Layer 1 (Ultra-Dark Blue) */}
      <div className="absolute inset-0 animate-pulse" 
           style={{ 
             background: 'radial-gradient(circle at 15% 15%, #050a1b 0%, transparent 70%), radial-gradient(circle at 85% 85%, #02071d 0%, transparent 70%)',
             filter: 'blur(100px)'
           }} />
      
      {/* 🌌 Solid Deep Nebula Layer 2 (Midnight Indigo) */}
      <div className="absolute inset-0 animate-slow-spin-slow" 
           style={{ 
             background: 'radial-gradient(circle at 70% 30%, #03041a 0%, transparent 60%), radial-gradient(circle at 30% 70%, #010214 0%, transparent 60%)',
             filter: 'blur(60px)'
           }} />

      {/* ⭐ Solid Parallax Stars (Subtle Glow) */}
      <div className="absolute inset-0" 
           style={{ 
             backgroundImage: 'radial-gradient(1px 1px at 20px 30px, #3b82f6, rgba(0,0,0,0)), radial-gradient(1px 1px at 150px 100px, #1e3a8a, rgba(0,0,0,0)), radial-gradient(1px 1px at 300px 450px, #60a5fa, rgba(0,0,0,0)), radial-gradient(1px 1px at 500px 300px, #fff, rgba(0,0,0,0)), radial-gradient(1px 1px at 700px 150px, #1e40af, rgba(0,0,0,0))',
             backgroundSize: '400px 400px',
             backgroundRepeat: 'repeat',
             animation: 'move-stars 250s linear infinite'
           }} />

      {/* ⭐ Sparse Bright Stars */}
      <div className="absolute inset-0 opacity-40" 
           style={{ 
             backgroundImage: 'radial-gradient(1.5px 1.5px at 100px 50px, #fff, rgba(0,0,0,0)), radial-gradient(1.5px 1.5px at 250px 200px, #fff, rgba(0,0,0,0)), radial-gradient(1.5px 1.5px at 450px 350px, #fff, rgba(0,0,0,0))',
             backgroundSize: '600px 600px',
             backgroundRepeat: 'repeat',
             animation: 'move-stars 180s linear infinite'
           }} />

      {/* 🌫️ Stardust Fog Pattern */}
      <div className="absolute inset-0 opacity-[0.03] mix-blend-lighten pointer-events-none" 
           style={{ background: 'url("https://www.transparenttextures.com/patterns/stardust.png") repeat' }} />

      <style jsx>{`
        @keyframes move-stars {
          from { transform: translateY(0); }
          to { transform: translateY(-1000px); }
        }
        @keyframes slow-spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-slow-spin-slow {
          animation: slow-spin-slow 200s linear infinite;
        }
      `}</style>
    </div>
  );
};

export default GalaxyBackground;
