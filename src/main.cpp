#include <SDL2/SDL.h>
#include <iostream>
#include <cmath>
#include "mandelbrot.cuh"

int WIDTH = 1280;
int HEIGHT = 720;
const int SUPERSAMPLE = 2;  // 2x supersampling for HD quality
const int MAX_ITER = 500;   // Full quality iterations
const int PREVIEW_ITER = 100; // Fast preview during interaction
const int PREVIEW_SCALE = 2;  // Render at half resolution during drag

double centerX = -0.5;
double centerY = 0.0;
double zoom = 1.0;

double prev_centerX = centerX;
double prev_centerY = centerY;
double prev_zoom = zoom;

bool isDragging = false;
bool isZooming = false;
bool needsRedraw = false;
bool needsRefine = false;
int dragStartX = 0;
int dragStartY = 0;
double dragStartCenterX = 0.0;
double dragStartCenterY = 0.0;

// Color table pointer (initialized in main)
ColorTable* colorTable = nullptr;


int main(int argc, char* argv[]) {

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        system("pause");
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "Example",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        WIDTH, HEIGHT,
        SDL_WINDOW_SHOWN
    );

    if ( !window ) {
        std::cout << "Failed to create a window! Error: " << SDL_GetError() << std::endl;
    }
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    if ( !renderer ) {
		std::cout << "Error creating renderer: " << SDL_GetError() << std::endl;
		return 1;
	}
    
    SDL_Texture* texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STREAMING,
        WIDTH, HEIGHT
    );

    if ( !texture ) {
        std::cout << "Error creating texture: " << SDL_GetError() << std::endl;
        return 1;
    }

    // Allocate pixel buffers
    // Supersampled buffer for HD rendering
    int ssWidth = WIDTH * SUPERSAMPLE;
    int ssHeight = HEIGHT * SUPERSAMPLE;
    uint8_t* ssPixels = new uint8_t[ssWidth * ssHeight * 4];
    uint8_t* pixels = new uint8_t[WIDTH * HEIGHT * 4];
    
    // Create cool color palette: deep blues, bright cyans, and rich purples
    colorTable = createSinColorTable(4096);
    
    // Initial computation with supersampling
    computeMandelbrotCUDA(ssPixels, ssWidth, ssHeight, centerX, centerY, zoom, MAX_ITER, colorTable);
    
    // Downsample for antialiasing
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int r = 0, g = 0, b = 0;
            // Average 2x2 supersampled pixels
            for (int sy = 0; sy < SUPERSAMPLE; sy++) {
                for (int sx = 0; sx < SUPERSAMPLE; sx++) {
                    int ssIdx = ((y * SUPERSAMPLE + sy) * ssWidth + (x * SUPERSAMPLE + sx)) * 4;
                    r += ssPixels[ssIdx + 0];
                    g += ssPixels[ssIdx + 1];
                    b += ssPixels[ssIdx + 2];
                }
            }
            int samples = SUPERSAMPLE * SUPERSAMPLE;
            int idx = (y * WIDTH + x) * 4;
            pixels[idx + 0] = r / samples;
            pixels[idx + 1] = g / samples;
            pixels[idx + 2] = b / samples;
            pixels[idx + 3] = 255;
        }
    }
    
    SDL_UpdateTexture(texture, NULL, pixels, WIDTH * 4);
    prev_centerX = centerX;
    prev_centerY = centerY;
    prev_zoom = zoom;
    
    // Event variable
    SDL_Event event;
    
    // Timing for refinement
    Uint32 lastInteractionTime = SDL_GetTicks();
    const Uint32 REFINE_DELAY = 150; // ms to wait before refining

    // Main loop
    bool running = true;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_MOUSEWHEEL) {
                if (event.wheel.y > 0) {
                    zoom *= 1.1; // Zoom in
                } else if (event.wheel.y < 0) {
                    zoom /= 1.1; // Zoom out
                }
                isZooming = true;
                needsRedraw = true;
                needsRefine = true;
                lastInteractionTime = SDL_GetTicks();
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    isDragging = true;
                    dragStartX = event.button.x;
                    dragStartY = event.button.y;
                    dragStartCenterX = centerX;
                    dragStartCenterY = centerY;
                }
            } else if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    isDragging = false;
                    needsRefine = true;
                    lastInteractionTime = SDL_GetTicks();
                }
            } else if (event.type == SDL_MOUSEMOTION) {
                if (isDragging) {
                    // Calculate drag delta in screen coordinates
                    int deltaX = event.motion.x - dragStartX;
                    int deltaY = event.motion.y - dragStartY;
                    
                    // Convert to complex plane coordinates and update center
                    centerX = dragStartCenterX - (deltaX / (WIDTH/2.0)) * (1.5 / zoom);
                    centerY = dragStartCenterY - (deltaY / (HEIGHT/2.0)) * (1.0 / zoom);
                    needsRedraw = true;
                    lastInteractionTime = SDL_GetTicks();
                }
            }
        }
        
        bool changed = (centerX != prev_centerX || centerY != prev_centerY || zoom != prev_zoom);
        // Reset zooming flag if enough time has passed
        if (isZooming && (SDL_GetTicks() - lastInteractionTime) > 50) {
            isZooming = false;
        }
        
        if (changed && needsRedraw) {
            if (isDragging || isZooming) {
                // Fast preview rendering during interaction
                // Fast preview rendering during drag - lower resolution and iterations
                int previewWidth = WIDTH / PREVIEW_SCALE;
                int previewHeight = HEIGHT / PREVIEW_SCALE;
                uint8_t* previewPixels = new uint8_t[previewWidth * previewHeight * 4];
                
                // Scale iterations with zoom to maintain detail when zoomed in
                // At zoom=1, use PREVIEW_ITER (100). At higher zoom, scale proportionally
                int adaptiveIter = std::min((int)(PREVIEW_ITER * log2(zoom + 1) * 0.7 + PREVIEW_ITER), MAX_ITER);
                
                computeMandelbrotCUDA(previewPixels, previewWidth, previewHeight, 
                                    centerX, centerY, zoom, adaptiveIter, colorTable);
                
                // Upscale preview to display buffer with nearest neighbor
                for (int y = 0; y < HEIGHT; y++) {
                    for (int x = 0; x < WIDTH; x++) {
                        int px = x / PREVIEW_SCALE;
                        int py = y / PREVIEW_SCALE;
                        int srcIdx = (py * previewWidth + px) * 4;
                        int dstIdx = (y * WIDTH + x) * 4;
                        pixels[dstIdx + 0] = previewPixels[srcIdx + 0];
                        pixels[dstIdx + 1] = previewPixels[srcIdx + 1];
                        pixels[dstIdx + 2] = previewPixels[srcIdx + 2];
                        pixels[dstIdx + 3] = 255;
                    }
                }
                delete[] previewPixels;
            } else {
                // Full quality rendering when not interacting
                computeMandelbrotCUDA(ssPixels, ssWidth, ssHeight, centerX, centerY, zoom, MAX_ITER, colorTable);
                
                // Downsample for antialiasing
                for (int y = 0; y < HEIGHT; y++) {
                    for (int x = 0; x < WIDTH; x++) {
                        int r = 0, g = 0, b = 0;
                        for (int sy = 0; sy < SUPERSAMPLE; sy++) {
                            for (int sx = 0; sx < SUPERSAMPLE; sx++) {
                                int ssIdx = ((y * SUPERSAMPLE + sy) * ssWidth + (x * SUPERSAMPLE + sx)) * 4;
                                r += ssPixels[ssIdx + 0];
                                g += ssPixels[ssIdx + 1];
                                b += ssPixels[ssIdx + 2];
                            }
                        }
                        int samples = SUPERSAMPLE * SUPERSAMPLE;
                        int idx = (y * WIDTH + x) * 4;
                        pixels[idx + 0] = r / samples;
                        pixels[idx + 1] = g / samples;
                        pixels[idx + 2] = b / samples;
                        pixels[idx + 3] = 255;
                    }
                }
                needsRefine = false;
            }
            
            SDL_UpdateTexture(texture, NULL, pixels, WIDTH * 4);
            prev_centerX = centerX;
            prev_centerY = centerY;
            prev_zoom = zoom;
            needsRedraw = false;
        } else if (needsRefine && !isDragging && !isZooming && 
                    (SDL_GetTicks() - lastInteractionTime) > REFINE_DELAY) {
            // Refine to full quality after interaction stops
            computeMandelbrotCUDA(ssPixels, ssWidth, ssHeight, centerX, centerY, zoom, MAX_ITER, colorTable);
            
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    int r = 0, g = 0, b = 0;
                    for (int sy = 0; sy < SUPERSAMPLE; sy++) {
                        for (int sx = 0; sx < SUPERSAMPLE; sx++) {
                            int ssIdx = ((y * SUPERSAMPLE + sy) * ssWidth + (x * SUPERSAMPLE + sx)) * 4;
                            r += ssPixels[ssIdx + 0];
                            g += ssPixels[ssIdx + 1];
                            b += ssPixels[ssIdx + 2];
                        }
                    }
                    int samples = SUPERSAMPLE * SUPERSAMPLE;
                    int idx = (y * WIDTH + x) * 4;
                    pixels[idx + 0] = r / samples;
                    pixels[idx + 1] = g / samples;
                    pixels[idx + 2] = b / samples;
                    pixels[idx + 3] = 255;
                }
            }
            
            SDL_UpdateTexture(texture, NULL, pixels, WIDTH * 4);
            needsRefine = false;
        }
        

        SDL_RenderClear(renderer);
        
        SDL_RenderCopy(renderer, texture, NULL, NULL);        

        SDL_RenderPresent(renderer);
    }
    
    // Cleanup
    if (colorTable) {
        freeColorTable(colorTable);
    }
    delete[] pixels;
    delete[] ssPixels;
    freeMandelbrotCudaResources();
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}