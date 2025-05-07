using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OnnxEmbeddings
{
    public class Clipboard()
    {
        [DllImport("user32.dll")]
        static extern bool OpenClipboard(IntPtr hWndNewOwner);

        [DllImport("user32.dll")]
        static extern bool CloseClipboard();

        [DllImport("user32.dll")]
        static extern IntPtr GetClipboardData(uint uFormat);

        [DllImport("kernel32.dll")]
        static extern IntPtr GlobalLock(IntPtr hMem);

        [DllImport("kernel32.dll")]
        static extern bool GlobalUnlock(IntPtr hMem);

        [DllImport("gdi32.dll")]
        private static extern bool DeleteObject(IntPtr hObject);

        [DllImport("gdi32.dll")]
        private static extern int GetObject(IntPtr hObject, int nSize, ref BITMAP lpBitmap);

        [DllImport("gdi32.dll")]
        private static extern IntPtr CreateCompatibleDC(IntPtr hdc);

        [DllImport("gdi32.dll")]
        private static extern IntPtr SelectObject(IntPtr hdc, IntPtr hgdiobj);

        [DllImport("gdi32.dll")]
        private static extern bool BitBlt(IntPtr hdcDest, int nXDest, int nYDest, int nWidth, int nHeight, IntPtr hdcSrc, int nXSrc, int nYSrc, uint dwRop);

        [DllImport("gdi32.dll")]
        private static extern IntPtr CreateCompatibleBitmap(IntPtr hdc, int nWidth, int nHeight);

        [DllImport("gdi32.dll")]
        private static extern bool DeleteDC(IntPtr hdc);
        const uint CF_UNICODETEXT = 13;
        public static string GetText()
        {
            string clipboardText = "";
            if (OpenClipboard(IntPtr.Zero))
            {
                try
                {
                    IntPtr hGlobal = GetClipboardData(CF_UNICODETEXT);
                    if (hGlobal != IntPtr.Zero)
                    {
                        IntPtr lpwcstr = GlobalLock(hGlobal);
                        if (lpwcstr != IntPtr.Zero)
                        {
                            try
                            {
                                clipboardText = Marshal.PtrToStringUni(lpwcstr);

                            }
                            finally
                            {
                                GlobalUnlock(hGlobal);
                            }
                        }
                        else
                        {
                            Console.WriteLine("Failed to lock the memory.");
                        }
                    }
                    else
                    {
                        Console.WriteLine("No text in clipboard.");
                    }
                }
                finally
                {
                    CloseClipboard();
                }
                return clipboardText;

            }
            else
            {
                return null;
            }
        }
        private const uint CF_BITMAP = 2;
        private const uint SRCCOPY = 0x00CC0020;
        [StructLayout(LayoutKind.Sequential)]
        private struct BITMAP
        {
            public int bmType;
            public int bmWidth;
            public int bmHeight;
            public int bmWidthBytes;
            public short bmPlanes;
            public short bmBitsPixel;
            public IntPtr bmBits;
        }
        [StructLayout(LayoutKind.Sequential)]
        private struct BITMAPINFOHEADER
        {
            public int biSize;
            public int biWidth;
            public int biHeight;
            public short biPlanes;
            public short biBitCount;
            public int biCompression;
            public int biSizeImage;
            public int biXPelsPerMeter;
            public int biYPelsPerMeter;
            public int biClrUsed;
            public int biClrImportant;
        }
    }
}
