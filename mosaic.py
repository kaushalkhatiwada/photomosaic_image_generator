import os
import cv2
import numpy as np

class PhotoMosaic:
    def __init__(self, original_image_path, root_path, tile_size,tile_folder):
        self.original_image_path = original_image_path
        self.original_image = self.load_image(self.original_image_path)         # 1) loading Original image
        self.tile_image_dir = os.path.join(root_path, tile_folder)
        self.tile_size = tile_size
        self.tiles = self.load_tiles()                                          # Loading tile images                 
        self.final_image = None

    # Load original image
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return image

    # Resize tile Image size
    def resize_image(self, image, size):
        return cv2.resize(image, size)

    # Get average color of Orifinal Image
    def get_average_color(self, image):
        return image.mean(axis=(0, 1))

    # Get average color of every tile image
    def load_tiles(self):
        tiles = []
        for image_name in os.listdir(self.tile_image_dir):
            image_path = os.path.join(self.tile_image_dir, image_name)
            tile_image = self.load_image(image_path)                                # Load Tile Image
            tile_image = self.resize_image(tile_image, self.tile_size)              # Resize Image
            tile_avg_color = self.get_average_color(tile_image)                     # Get Average Color
            tiles.append((tile_avg_color, tile_image))
        return tiles

    # Create Grid of Original Image
    def create_grid(self):
        original_image_shape = self.original_image.shape
        grid_size = (original_image_shape[1] // self.tile_size[0], original_image_shape[0] // self.tile_size[1])
        return grid_size

    # Find the Best Tile Image using euclidean distance between the average color of tile and average color of the grid region.
    def find_best_tile(self, region_avg_color):
        min_distance = float('inf')
        best_tile = None
        for tile_avg_color, tile_image in self.tiles:
            distance = np.linalg.norm(tile_avg_color - region_avg_color)
            if distance < min_distance:
                min_distance = distance
                best_tile = tile_image
        return best_tile

    # Create Mosaic Image by replacing the best fit tile to the grid
    def create_mosaic(self):
        grid_size = self.create_grid()                                  # Creating Grid of an Original Image
        grid_row_size, grid_column_size = grid_size
        self.final_image = np.zeros((grid_column_size * self.tile_size[1], grid_row_size * self.tile_size[0], 3), dtype=np.uint8)

        for i in range(grid_row_size):
            for j in range(grid_column_size):
                region = self.original_image[j * self.tile_size[1]:(j + 1) * self.tile_size[1], i * self.tile_size[0]:(i + 1) * self.tile_size[0]]
                region_avg_color = self.get_average_color(region)           # Get average color
                best_tile = self.find_best_tile(region_avg_color)           # Finding Best tile
                self.final_image[j * self.tile_size[1]:(j + 1) * self.tile_size[1], i * self.tile_size[0]:(i + 1) * self.tile_size[0]] = best_tile

    # Save Mosaic Image
    def save_mosaic(self, output_path):
        final_image_bgr = cv2.cvtColor(self.final_image, cv2.COLOR_RGB2BGR) 
        cv2.imwrite(output_path, final_image_bgr)
        print("Mosiac Image created")

    # def display_mosaic(self):
    #     cv2.imshow("Final Image", cv2.cvtColor(self.final_image, cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

def main():
    dir = os.path.dirname(os.path.abspath(__file__))
    original_image_path = os.path.join(dir, 'kk.jpg')
    output_path = os.path.join(dir, f'{original_image_path}_mosiac.jpg')
    tile_size=(20, 20)
    tile_folder='images'

    photomosaic = PhotoMosaic(original_image_path, dir,tile_size,tile_folder)    # Creating Object
    photomosaic.create_mosaic()                                                         # Calling "create" method
    photomosaic.save_mosaic(output_path)                                                # Calling "save" method
    # photomosaic.display_mosaic() 

if __name__ == "__main__":
    main()