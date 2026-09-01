/**
 * Professional Diagnostic Document Generator
 * Creates structured maintenance/troubleshooting reports
 */

import { jsPDF } from 'jspdf';

export interface DiagnosticReport {
  sessionId: number;
  machineId?: string;
  problemDescription: string;
  diagnosis: string;
  solutionSteps: string[];
  images: Array<{ url: string; caption?: string }>;
  timestamp: string;
  operatorNotes?: string;
}

// Professional color scheme
const COLORS = {
  primary: [25, 118, 210] as [number, number, number],      // Blue
  headerBg: [13, 71, 161] as [number, number, number],      // Dark Blue
  accent: [16, 185, 129] as [number, number, number],       // Green
  text: [31, 41, 55] as [number, number, number],           // Dark Gray
  lightGray: [243, 244, 246] as [number, number, number],   // Light Gray bg
  border: [209, 213, 219] as [number, number, number],      // Border Gray
  white: [255, 255, 255] as [number, number, number],
};

// Layout constants
const MARGIN = {
  left: 20,
  right: 20,
  top: 20,
  bottom: 25,
};
const PAGE_WIDTH = 210;
const CONTENT_WIDTH = PAGE_WIDTH - MARGIN.left - MARGIN.right;

/**
 * Convert image URL to base64
 */
async function imageToBase64(url: string): Promise<string> {
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch (error) {
    console.error('Failed to load image:', error);
    return '';
  }
}

/**
 * Add diagonal watermark to all pages
 */
function addWatermark(doc: jsPDF) {
  const pageCount = doc.internal.pages.length - 1;
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();

  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.saveGraphicsState();
    doc.setGState(doc.GState({ opacity: 0.06 }));
    doc.setTextColor(COLORS.primary[0], COLORS.primary[1], COLORS.primary[2]);
    doc.setFontSize(70);
    doc.setFont('helvetica', 'bold');
    doc.text('ZYNAPTRIX', pageWidth / 2, pageHeight / 2, {
      align: 'center',
      angle: 45,
    });
    doc.restoreGraphicsState();
  }
}

/**
 * Add professional header with logo area
 */
function addHeader(doc: jsPDF, report: DiagnosticReport): number {
  // Header background
  doc.setFillColor(COLORS.headerBg[0], COLORS.headerBg[1], COLORS.headerBg[2]);
  doc.rect(0, 0, PAGE_WIDTH, 40, 'F');
  
  // Title
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(26);
  doc.setFont('helvetica', 'bold');
  doc.text('DIAGNOSTIC REPORT', MARGIN.left, 22);
  
  // Subtitle
  doc.setFontSize(11);
  doc.setFont('helvetica', 'normal');
  doc.text('Zynaptrix Industrial Copilot Platform', MARGIN.left, 32);
  
  // Company branding on right
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text('ZYNAPTRIX', PAGE_WIDTH - MARGIN.right, 22, { align: 'right' });
  doc.setFontSize(8);
  doc.setFont('helvetica', 'normal');
  doc.text('Industrial AI Solutions', PAGE_WIDTH - MARGIN.right, 30, { align: 'right' });
  
  // Info box below header
  let yPos = 50;
  doc.setFillColor(COLORS.lightGray[0], COLORS.lightGray[1], COLORS.lightGray[2]);
  doc.setDrawColor(COLORS.border[0], COLORS.border[1], COLORS.border[2]);
  doc.setLineWidth(0.3);
  doc.roundedRect(MARGIN.left, yPos, CONTENT_WIDTH, 28, 3, 3, 'FD');
  
  // Info labels and values
  doc.setFontSize(9);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(COLORS.text[0], COLORS.text[1], COLORS.text[2]);
  
  const col1 = MARGIN.left + 8;
  const col2 = 55;
  const col3 = 115;
  const col4 = 145;
  
  doc.text('Report ID:', col1, yPos + 10);
  doc.text('Machine:', col1, yPos + 18);
  doc.text('Date:', col3, yPos + 10);
  doc.text('Status:', col3, yPos + 18);
  
  doc.setFont('helvetica', 'normal');
  doc.text(`#${report.sessionId}`, col2, yPos + 10);
  const machineText = report.machineId || 'N/A';
  doc.text(machineText.length > 25 ? machineText.substring(0, 25) + '...' : machineText, col2, yPos + 18);
  doc.text(new Date(report.timestamp).toLocaleDateString(), col4, yPos + 10);
  
  // Status badge
  doc.setFillColor(COLORS.accent[0], COLORS.accent[1], COLORS.accent[2]);
  doc.roundedRect(col4, yPos + 13, 22, 6, 2, 2, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(7);
  doc.setFont('helvetica', 'bold');
  doc.text('COMPLETE', col4 + 11, yPos + 17.5, { align: 'center' });
  
  return yPos + 38;
}

/**
 * Add section header with colored bar
 */
function addSectionHeader(doc: jsPDF, title: string, yPos: number): number {
  const pageHeight = doc.internal.pageSize.getHeight();
  
  // Check for page break
  if (yPos > pageHeight - 50) {
    doc.addPage();
    yPos = MARGIN.top;
  }
  
  // Left accent bar
  doc.setFillColor(COLORS.primary[0], COLORS.primary[1], COLORS.primary[2]);
  doc.rect(MARGIN.left, yPos, 4, 8, 'F');
  
  // Section background
  doc.setFillColor(COLORS.lightGray[0], COLORS.lightGray[1], COLORS.lightGray[2]);
  doc.rect(MARGIN.left + 4, yPos, CONTENT_WIDTH - 4, 8, 'F');
  
  // Section title
  doc.setTextColor(COLORS.text[0], COLORS.text[1], COLORS.text[2]);
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  doc.text(title, MARGIN.left + 10, yPos + 5.5);
  
  return yPos + 14;
}

/**
 * Add text content with proper word wrap
 */
function addTextSection(doc: jsPDF, content: string, yPos: number): number {
  const pageHeight = doc.internal.pageSize.getHeight();
  
  doc.setTextColor(COLORS.text[0], COLORS.text[1], COLORS.text[2]);
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  
  const lines = doc.splitTextToSize(content, CONTENT_WIDTH - 10) as string[];
  const lineHeight = 5;
  
  for (const line of lines) {
    if (yPos > pageHeight - MARGIN.bottom - 10) {
      doc.addPage();
      yPos = MARGIN.top;
    }
    doc.text(line, MARGIN.left + 5, yPos);
    yPos += lineHeight;
  }
  
  return yPos + 5;
}

/**
 * Add numbered steps with inline professional styling and smaller images
 */
async function addStepsList(doc: jsPDF, steps: string[], images: Array<{url: string; caption?: string}>, yPos: number): Promise<number> {
  const pageHeight = doc.internal.pageSize.getHeight();
  
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    const stepNum = i + 1;
    
    // Check page break - need space for step
    if (yPos > pageHeight - MARGIN.bottom - 20) {
      doc.addPage();
      yPos = MARGIN.top;
    }
    
    // Step number circle
    const circleX = MARGIN.left + 8;
    const circleY = yPos;
    doc.setFillColor(COLORS.accent[0], COLORS.accent[1], COLORS.accent[2]);
    doc.circle(circleX, circleY, 4, 'F');
    
    // Step number text
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.text(String(stepNum), circleX, circleY + 1, { align: 'center' });
    
    // Step content
    doc.setTextColor(COLORS.text[0], COLORS.text[1], COLORS.text[2]);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    
    const textX = MARGIN.left + 18;
    const maxTextWidth = CONTENT_WIDTH - 25;
    const lines = doc.splitTextToSize(step, maxTextWidth) as string[];
    
    let firstLine = true;
    for (const line of lines) {
      if (yPos > pageHeight - MARGIN.bottom - 10) {
        doc.addPage();
        yPos = MARGIN.top;
      }
      doc.text(line, textX, firstLine ? yPos + 1 : yPos);
      yPos += 5;
      firstLine = false;
    }
    
    yPos += 4; // Space after step text

    // Check if we should inject an image under this step
    // Simply map image `i` to step `i`
    if (images && images.length > i) {
      const img = images[i];
      if (yPos > pageHeight - 65) {
        doc.addPage();
        yPos = MARGIN.top;
      }
      try {
        const base64Image = await imageToBase64(img.url);
        if (base64Image) {
          // Smaller, professional inline image layout
          const imgWidth = 80;
          const imgHeight = 45;
          const imgX = MARGIN.left + 18; // Align with step text
          
          doc.setDrawColor(COLORS.border[0], COLORS.border[1], COLORS.border[2]);
          doc.setLineWidth(0.3);
          doc.roundedRect(imgX, yPos, imgWidth, imgHeight, 2, 2, 'S');
          
          let format = 'JPEG';
          if (base64Image.startsWith('data:image/png')) format = 'PNG';
          else if (base64Image.startsWith('data:image/webp')) format = 'WEBP';

          doc.addImage(base64Image, format, imgX + 1, yPos + 1, imgWidth - 2, imgHeight - 2);
          yPos += imgHeight + 3;
          
          if (img.caption) {
            doc.setFontSize(7);
            doc.setTextColor(120, 120, 120);
            doc.setFont('helvetica', 'italic');
            doc.text(img.caption, imgX + imgWidth / 2, yPos, { align: 'center' });
            yPos += 4;
          }
          yPos += 4; // Extra padding after image
        }
      } catch (err) {
        console.error('Failed to inline image', err);
      }
    }
  }

  // If there are leftover images (more images than steps), dump them at the end of the list
  if (images && images.length > steps.length) {
    for (let i = steps.length; i < images.length; i++) {
      const img = images[i];
      if (yPos > pageHeight - 65) {
        doc.addPage();
        yPos = MARGIN.top;
      }
      try {
        const base64Image = await imageToBase64(img.url);
        if (base64Image) {
          const imgWidth = 80;
          const imgHeight = 45;
          const imgX = MARGIN.left + 18;
          
          doc.setDrawColor(COLORS.border[0], COLORS.border[1], COLORS.border[2]);
          doc.setLineWidth(0.3);
          doc.roundedRect(imgX, yPos, imgWidth, imgHeight, 2, 2, 'S');
          
          let format = 'JPEG';
          if (base64Image.startsWith('data:image/png')) format = 'PNG';
          else if (base64Image.startsWith('data:image/webp')) format = 'WEBP';

          doc.addImage(base64Image, format, imgX + 1, yPos + 1, imgWidth - 2, imgHeight - 2);
          yPos += imgHeight + 3;
          if (img.caption) {
            doc.setFontSize(7);
            doc.setTextColor(120, 120, 120);
            doc.setFont('helvetica', 'italic');
            doc.text(img.caption, imgX + imgWidth / 2, yPos, { align: 'center' });
            yPos += 4;
          }
          yPos += 4;
        }
      } catch (e) {}
    }
  }
  
  return yPos + 5;
}

/**
 * Add images with captions
 */
async function addImages(doc: jsPDF, images: Array<{ url: string; caption?: string }>, yPos: number): Promise<number> {
  const pageHeight = doc.internal.pageSize.getHeight();
  
  for (const img of images) {
    // Check if need new page
    if (yPos > pageHeight - 100) {
      doc.addPage();
      yPos = MARGIN.top;
    }
    
    try {
      const base64Image = await imageToBase64(img.url);
      if (base64Image) {
        const imgWidth = CONTENT_WIDTH - 20;
        const imgHeight = 70;
        
        // Image border
        doc.setDrawColor(COLORS.border[0], COLORS.border[1], COLORS.border[2]);
        doc.setLineWidth(0.5);
        doc.roundedRect(MARGIN.left + 10, yPos, imgWidth, imgHeight, 2, 2, 'S');
        
        // Ensure correct format for jsPDF (it fails if you pass JPEG but the header is PNG)
        let format = 'JPEG';
        if (base64Image.startsWith('data:image/png')) format = 'PNG';
        else if (base64Image.startsWith('data:image/webp')) format = 'WEBP';

        // Image
        doc.addImage(base64Image, format, MARGIN.left + 12, yPos + 2, imgWidth - 4, imgHeight - 4);
        yPos += imgHeight + 3;
        
        // Caption
        if (img.caption) {
          doc.setFontSize(8);
          doc.setTextColor(100, 100, 100);
          doc.setFont('helvetica', 'italic');
          doc.text(img.caption, MARGIN.left + 10 + imgWidth / 2, yPos, { align: 'center' });
          yPos += 6;
        }
        
        yPos += 8;
      }
    } catch (error) {
      console.error('Failed to add image:', error);
      doc.setFontSize(9);
      doc.setTextColor(150, 150, 150);
      doc.text('[Image could not be loaded]', MARGIN.left + 10, yPos);
      yPos += 10;
    }
  }
  
  return yPos;
}

/**
 * Add footer to all pages
 */
function addFooters(doc: jsPDF) {
  const pageCount = doc.internal.pages.length - 1;
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    
    // Footer line
    doc.setDrawColor(COLORS.border[0], COLORS.border[1], COLORS.border[2]);
    doc.setLineWidth(0.3);
    doc.line(MARGIN.left, pageHeight - 18, PAGE_WIDTH - MARGIN.right, pageHeight - 18);
    
    // Footer text
    doc.setFontSize(8);
    doc.setTextColor(120, 120, 120);
    doc.setFont('helvetica', 'normal');
    
    // Left: confidential
    doc.text('CONFIDENTIAL - Internal Use Only', MARGIN.left, pageHeight - 12);
    
    // Center: generated by
    doc.text('Generated by Zynaptrix Industrial Copilot', pageWidth / 2, pageHeight - 12, { align: 'center' });
    
    // Right: page number
    doc.text(`Page ${i} of ${pageCount}`, PAGE_WIDTH - MARGIN.right, pageHeight - 12, { align: 'right' });
  }
}

/**
 * Generate professional diagnostic report PDF
 */
export async function generateDiagnosticReport(
  report: DiagnosticReport,
  onProgress?: (progress: number) => void
): Promise<void> {
  try {
    const doc = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
    });
    
    if (onProgress) onProgress(10);
    
    // Header
    let yPos = addHeader(doc, report);
    yPos += 8;
    
    if (onProgress) onProgress(20);
    
    // Problem Description
    yPos = addSectionHeader(doc, 'PROBLEM DESCRIPTION', yPos);
    yPos = addTextSection(doc, report.problemDescription || 'No problem description provided.', yPos);
    yPos += 8;
    
    if (onProgress) onProgress(35);
    
    // Diagnosis
    yPos = addSectionHeader(doc, 'DIAGNOSIS', yPos);
    yPos = addTextSection(doc, report.diagnosis || 'No diagnosis available.', yPos);
    yPos += 8;
    
    if (onProgress) onProgress(50);
    
    // Solution Steps + Inline Images
    if (report.solutionSteps && report.solutionSteps.length > 0) {
      yPos = addSectionHeader(doc, 'SOLUTION / REPAIR PROCEDURE', yPos);
      yPos = await addStepsList(doc, report.solutionSteps, report.images || [], yPos);
      yPos += 8;
    } else if (report.images && report.images.length > 0) {
      // Fallback if no steps but images exist (rare)
      yPos = addSectionHeader(doc, 'REFERENCE DIAGRAMS', yPos);
      yPos = await addImages(doc, report.images, yPos);
    }
    
    if (onProgress) onProgress(85);
    
    // Operator Notes
    if (report.operatorNotes) {
      yPos = addSectionHeader(doc, 'OPERATOR NOTES', yPos);
      yPos = addTextSection(doc, report.operatorNotes, yPos);
    }
    
    // Add watermark and footers
    addWatermark(doc);
    addFooters(doc);
    
    if (onProgress) onProgress(95);
    
    // Save file
    const dateStr = new Date().toISOString().split('T')[0];
    const filename = `Diagnostic_Report_${report.sessionId}_${dateStr}.pdf`;
    doc.save(filename);
    
    if (onProgress) onProgress(100);
  } catch (error) {
    console.error('Failed to generate report:', error);
    throw new Error('PDF generation failed');
  }
}
