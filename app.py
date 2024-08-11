import streamlit as st
import cv2
import numpy as np
import os

class PhotoMosaic:
    def __init__(self, original_image_path, tile_image_dir, tile_size):
        self.original_image_path = original_image_path
        self.original_image = self.load_image(self.original_image_path)  # 1) loading Original image
        self.tile_image_dir = tile_image_dir
        self.tile_size = tile_size
        self.tiles = self.load_tiles()                                   # 2) Loading tile images                 
        self.final_image = None

    # 1 Load original image
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return image

    # 2 Resize tile Image size
    def resize_image(self, image, size):
        return cv2.resize(image, size)

    # 3 Get average color of Original Image
    def get_average_color(self, image):
        return image.mean(axis=(0, 1))

    # 4 Get average color of every tile image
    def load_tiles(self):
        tiles = []
        for image_name in os.listdir(self.tile_image_dir):
            image_path = os.path.join(self.tile_image_dir, image_name)
            tile_image = self.load_image(image_path)  # Load Tile Image
            tile_image = self.resize_image(tile_image, self.tile_size)           # 3) Resize Tile Image
            tile_avg_color = self.get_average_color(tile_image)                  # 4) Get Average Color of tile image
            tiles.append((tile_avg_color, tile_image))
        return tiles

    # 5 Create Grid of Original Image
    def create_grid(self):
        original_image_shape = self.original_image.shape
        grid_size = (original_image_shape[1] // self.tile_size[0], original_image_shape[0] // self.tile_size[1])
        return grid_size

    # 6 Find the Best Tile Image using euclidean distance between the average color of tile and average color of the grid region.
    def find_best_tile(self, region_avg_color):
        min_distance = float('inf')
        best_tile = None
        for tile_avg_color, tile_image in self.tiles:
            distance = np.linalg.norm(tile_avg_color - region_avg_color)
            if distance < min_distance:
                min_distance = distance
                best_tile = tile_image
        return best_tile

    # 7 Create Mosaic Image by replacing the best fit tile to the grid
    def create_mosaic(self):
        grid_size = self.create_grid()                                      # 5) Creating Grid of an Original Image
        grid_row_size, grid_column_size = grid_size
        self.final_image = np.zeros((grid_column_size * self.tile_size[1], grid_row_size * self.tile_size[0], 3), dtype=np.uint8)

        for i in range(grid_row_size):
            for j in range(grid_column_size):
                region = self.original_image[j * self.tile_size[1]:(j + 1) * self.tile_size[1], i * self.tile_size[0]:(i + 1) * self.tile_size[0]]
                region_avg_color = self.get_average_color(region)      # 6) Get average color of Grid region of original image
                best_tile = self.find_best_tile(region_avg_color)      # 7) Finding Best tile
                self.final_image[j * self.tile_size[1]:(j + 1) * self.tile_size[1], i * self.tile_size[0]:(i + 1) * self.tile_size[0]] = best_tile

    # Save Mosaic Image
    def save_mosaic(self, output_path):
        final_image = cv2.cvtColor(self.final_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_image)

def main():
    st.title("Photo Mosaic Generator")

    # Upload original image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Set tile size
    tile_size = st.slider("Select tile size:", min_value=5, max_value=100, value=20)

    # Create image button
    if st.button("Create Image"):
        if uploaded_file is not None:
            # Ensure the uploads directory exists
            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)

            # Save the uploaded original image to the uploads directory
            original_image_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(original_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Get the directory of the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tile_image_dir = os.path.join(script_dir, 'images')

            # Ensure the mosaic directory exists
            mosaic_dir = os.path.join(os.getcwd(), 'mosaic')
            os.makedirs(mosaic_dir, exist_ok=True)

            # Create and process the photo mosaic
            photomosaic = PhotoMosaic(original_image_path, tile_image_dir, (tile_size, tile_size))   # Object created
            photomosaic.create_mosaic()

            # Save the final mosaic image to the mosaic directory
            output_path = os.path.join(mosaic_dir, f'mosaic_{uploaded_file.name}')
            photomosaic.save_mosaic(output_path)
            
            # Display the final mosaic image
            st.image(output_path, caption='Generated Photo Mosaic', use_column_width=True)

            # Provide download link for the final mosaic image
            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="Download Mosaic Image",
                    data=file,
                    file_name=f'mosaic_{uploaded_file.name}',
                    mime="image/jpeg"
                )
        else:
            st.error("Please upload an original image.")

if __name__ == "__main__":
    main()
