"use client"

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, Database, Cog, Factory, Menu, X } from 'lucide-react';

const NavBar = () => {
  const pathname = usePathname();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navItems = [
    { name: 'Dashboard', href: '/', icon: LayoutDashboard },
    { name: 'Machine Management', href: '/machines', icon: Cog },
    { name: 'Knowledge Ingestion', href: '/ingestion', icon: Database },
  ];

  return (
    <nav className={`${isMenuOpen ? 'bg-slate-950' : 'bg-slate-900/80 backdrop-blur-md'} border-b border-slate-800 px-4 md:px-8 py-4 sticky top-0 z-50 flex justify-between items-center shadow-2xl transition-colors duration-300`}>
      <Link href="/" className="flex items-center gap-3 group hover:opacity-80 transition-all cursor-pointer z-50">
        <div className="bg-blue-600 p-2 rounded-lg shadow-lg shadow-blue-500/20 group-hover:scale-110 transition-transform duration-300">
          <Factory className="text-white" size={20} />
        </div>
        <span className="text-xl font-black bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent tracking-tight">
          ZYNAPTRIX <span className="text-blue-500 font-light hidden xs:inline">COPILOT</span>
        </span>
      </Link>

      {/* Desktop Navigation */}
      <div className="hidden lg:flex items-center gap-2">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-300 text-sm font-bold tracking-wide ${
                isActive 
                  ? 'bg-blue-600/10 text-blue-400 border border-blue-500/30' 
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800'
              }`}
            >
              <Icon size={18} />
              {item.name}
            </Link>
          );
        })}
      </div>

      <div className="flex items-center gap-4 z-50">
        <div className="hidden lg:block h-4 w-[1px] bg-slate-800 mx-2"></div>
        <button className="hidden lg:block text-slate-400 hover:text-white transition-colors">
          <Cog size={20} />
        </button>
        
        {/* Mobile Menu Toggle */}
        <button 
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          className="lg:hidden p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-xl transition-all"
        >
          {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Menu Overlay */}
      {isMenuOpen && (
        <div className="absolute top-[calc(100%+0.5rem)] right-4 left-4 bg-slate-950 border border-slate-800 rounded-3xl shadow-[0_20px_50px_rgba(0,0,0,0.5)] z-40 lg:hidden animate-in fade-in slide-in-from-top-4 duration-300 overflow-hidden">
          <div className="flex flex-col p-6 gap-3">
            {navItems.map((item) => {
              const isActive = pathname === item.href;
              const Icon = item.icon;
              
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  onClick={() => setIsMenuOpen(false)}
                  className={`flex items-center gap-4 p-4 rounded-2xl transition-all duration-300 text-base font-bold tracking-wide ${
                    isActive 
                      ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30' 
                      : 'text-slate-400 border border-transparent active:bg-slate-800'
                  }`}
                >
                  <Icon size={22} />
                  {item.name}
                </Link>
              );
            })}
            <div className="mt-4 pt-4 border-t border-slate-800 flex justify-between items-center px-4">
               <span className="text-xs font-black text-slate-600 uppercase tracking-[0.2em]">Settings</span>
               <button className="text-slate-400">
                 <Cog size={20} />
               </button>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default NavBar;
