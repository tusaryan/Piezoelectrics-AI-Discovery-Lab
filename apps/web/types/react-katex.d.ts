declare module 'react-katex' {
  import React from 'react';
  interface MathProps {
    math?: string;
    children?: string;
    errorColor?: string;
    renderError?: (error: Error) => React.ReactNode;
  }
  export const InlineMath: React.FC<MathProps>;
  export const BlockMath: React.FC<MathProps>;
}
