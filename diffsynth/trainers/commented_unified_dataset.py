# ============================================================================
# EDUCATIONAL COMMENTS FOR PHASE 2 LEARNING - Data Loading System
# ============================================================================
"""
📚 COMMENTED VERSION OF diffsynth/trainers/unified_dataset.py

This file contains the DATA LOADING SYSTEM used for training all models in DiffSynth-Studio.
It provides a flexible, composable pipeline for loading and preprocessing training data.

📍 WHAT'S IN THIS FILE:
1. DataProcessingPipeline - Chains operators together (lines 40-59)
2. DataProcessingOperator - Base class for data transformations (lines 63-70)
3. Specific Operators - LoadImage, LoadVideo, ImageCropAndResize, etc. (lines 74-269)
4. UnifiedDataset - Main dataset class (lines 273-381)

🎯 WHY THIS MATTERS FOR S2V TRAINING:
- Training script uses UnifiedDataset to load RenderMe360 data
- Operators chain together: LoadVideo → Resize → Normalize
- Metadata (CSV/JSON) maps to actual data files
- Supports both image and video modalities

📖 READING ORDER:
1. DataProcessingOperator + DataProcessingPipeline (understand composition)
2. Specific operators (LoadImage, LoadVideo, etc.)
3. UnifiedDataset (how everything comes together)
4. Look at training script to see usage example

🔗 OPERATOR CHAINING WITH >> :
Operators can be chained using Python's >> operator:
    LoadImage() >> Resize(512, 512) >> ToTensor()
This creates a DataProcessingPipeline that applies operations sequentially.
"""

import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image


# ============================================================================
# CLASS 1: DataProcessingPipeline - Chain of Data Transformations
# ============================================================================
# PURPOSE: Container that applies multiple DataProcessingOperators sequentially
#
# HOW IT WORKS:
# - Stores list of operators
# - __call__() applies each operator in sequence
# - Supports chaining with >> operator
#
# EXAMPLE:
#   pipeline = LoadImage() >> Resize(512, 512) >> ToTensor()
#   result = pipeline("path/to/image.jpg")
#   # Equivalent to: ToTensor()(Resize(512,512)(LoadImage()("path/to/image.jpg")))
# ============================================================================

class DataProcessingPipeline:
    """
    Container for chaining multiple data processing operators.

    Applies operators sequentially: data → op1 → op2 → ... → opN
    """

    def __init__(self, operators=None):
        """
        Initialize pipeline with list of operators.

        Args:
            operators: List of DataProcessingOperator instances (default: empty list)
        """
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators  # Store list of operators to apply sequentially. If None provided, start with empty list

    def __call__(self, data):
        """
        Apply all operators in sequence to data.

        Process: data → operator[0] → operator[1] → ... → operator[N]

        Args:
            data: Input data (type depends on first operator)

        Returns:
            Transformed data (type depends on last operator)
        """
        for operator in self.operators:  # Iterate through each operator in the pipeline
            data = operator(data)  # Apply operator to data, update data with result. Each operator's output becomes next operator's input
        return data  # Return final transformed data after all operators applied

    def __rshift__(self, pipe):
        """
        Overload >> operator to chain pipelines/operators.

        Allows: pipeline1 >> pipeline2 or pipeline >> operator

        Args:
            pipe: DataProcessingPipeline or DataProcessingOperator to append

        Returns:
            New DataProcessingPipeline with combined operators
        """
        if isinstance(pipe, DataProcessingOperator):  # If pipe is a single operator (not a pipeline)
            pipe = DataProcessingPipeline([pipe])  # Wrap operator in a pipeline with one element. Now pipe is a DataProcessingPipeline with one operator
        return DataProcessingPipeline(self.operators + pipe.operators)  # Create new pipeline by concatenating operator lists. self.operators + pipe.operators combines both lists into a new list



# ============================================================================
# CLASS 2: DataProcessingOperator - Base Class for Data Transformations
# ============================================================================
# PURPOSE: Abstract base class for all data processing operators
#
# KEY FEATURES:
# - Must implement __call__() to process data
# - Supports >> operator for chaining
# - Subclasses implement specific transformations
#
# BUILT-IN OPERATORS:
# - ToInt, ToFloat, ToStr: Type conversions
# - LoadImage, LoadVideo: File loading
# - ImageCropAndResize: Preprocessing
# - ToAbsolutePath: Path manipulation
# ============================================================================

class DataProcessingOperator:
    """
    Base class for all data processing operators.

    Subclasses must implement __call__() to transform data.
    """

    def __call__(self, data):
        """
        Process data (must be overridden by subclasses).

        Args:
            data: Input data

        Returns:
            Transformed data
        """
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")  # This is abstract base class, subclasses must implement __call__()

    def __rshift__(self, pipe):
        """
        Overload >> operator to chain this operator with another.

        Allows: operator1 >> operator2 or operator >> pipeline

        Args:
            pipe: DataProcessingOperator or DataProcessingPipeline to chain

        Returns:
            New DataProcessingPipeline with both operators
        """
        if isinstance(pipe, DataProcessingOperator):  # If pipe is a single operator
            pipe = DataProcessingPipeline([pipe])  # Wrap it in a pipeline. Convert operator to pipeline with one element
        return DataProcessingPipeline([self]).__rshift__(pipe)  # Create pipeline from self, then chain with pipe using pipeline's __rshift__. Equivalent to: [self] >> pipe


# ============================================================================
# SIMPLE TYPE CONVERSION OPERATORS
# ============================================================================
# These operators convert data types without loading files
# Used for metadata fields (frame counts, dimensions, etc.)
# ============================================================================

class DataProcessingOperatorRaw(DataProcessingOperator):
    """
    Pass-through operator (returns data unchanged).

    Useful as placeholder or default operator.
    """

    def __call__(self, data):
        """Return data unchanged."""
        return data  # Identity function: output = input


class ToInt(DataProcessingOperator):
    """
    Convert data to integer.

    Example: "123" → 123
    """

    def __call__(self, data):
        """Convert data to int."""
        return int(data)  # Cast to integer type. Raises ValueError if data can't be converted


class ToFloat(DataProcessingOperator):
    """
    Convert data to float.

    Example: "3.14" → 3.14
    """

    def __call__(self, data):
        """Convert data to float."""
        return float(data)  # Cast to float type. Raises ValueError if data can't be converted


class ToStr(DataProcessingOperator):
    """
    Convert data to string, with optional default for None.

    Example: None → "" (if none_value="")
             123 → "123"
    """

    def __init__(self, none_value=""):
        """
        Initialize string converter.

        Args:
            none_value: String to use when data is None (default: "")
        """
        self.none_value = none_value  # Store default value for None inputs

    def __call__(self, data):
        """
        Convert data to string.

        Args:
            data: Input data (any type)

        Returns:
            String representation of data, or none_value if data is None
        """
        if data is None: data = self.none_value  # Replace None with default value. Prevents "None" string
        return str(data)  # Convert to string. Works for most Python types



# ============================================================================
# IMAGE LOADING AND PREPROCESSING OPERATORS
# ============================================================================
# These operators handle image files:
# - LoadImage: PIL.Image loading from file path
# - ImageCropAndResize: Smart resizing with aspect ratio preservation
# ============================================================================

class LoadImage(DataProcessingOperator):
    """
    Load image from file path using PIL.

    Handles various formats: JPG, PNG, WEBP, etc.
    Optionally converts to RGB (removes alpha channel).
    """

    def __init__(self, convert_RGB=True):
        """
        Initialize image loader.

        Args:
            convert_RGB: If True, convert image to RGB (removes alpha channel)
        """
        self.convert_RGB = convert_RGB  # Store whether to convert to RGB. True removes alpha channel for consistency

    def __call__(self, data: str):
        """
        Load image from file path.

        Args:
            data: File path to image (string)

        Returns:
            PIL.Image object
        """
        image = Image.open(data)  # Open image file using PIL. Returns PIL.Image object. Supports JPG, PNG, WEBP, etc.
        if self.convert_RGB: image = image.convert("RGB")  # Convert to RGB mode if requested. Removes alpha channel (RGBA→RGB), converts grayscale to RGB
        return image  # Return PIL.Image in RGB format


class ImageCropAndResize(DataProcessingOperator):
    """
    Smart image resizing with aspect ratio preservation and division constraints.

    TWO MODES:
    1. Fixed size: Resize to exact height×width
    2. Dynamic size: Resize to fit max_pixels, round to division factors

    PROCESS:
    - Calculate target size (respecting max_pixels)
    - Scale image to cover target (preserving aspect ratio)
    - Center crop to exact target size
    - Result: No distortion, meets dimension constraints

    EXAMPLE:
    - Input: 1920×1080 image
    - Target: max_pixels=1048576 (1024×1024), division_factor=16
    - Output: 1024×1024 image (scaled and cropped)
    """

    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        """
        Initialize image resizer.

        Args:
            height: Fixed height (None = dynamic)
            width: Fixed width (None = dynamic)
            max_pixels: Maximum total pixels (e.g., 1920*1080)
            height_division_factor: Height must be divisible by this (e.g., 16)
            width_division_factor: Width must be divisible by this (e.g., 16)
        """
        self.height = height  # Fixed height (None for dynamic). If provided, always resize to this height
        self.width = width  # Fixed width (None for dynamic). If provided, always resize to this width
        self.max_pixels = max_pixels  # Maximum resolution (total pixels). Used for dynamic sizing to limit image size
        self.height_division_factor = height_division_factor  # Height constraint (e.g., 16). Final height must be divisible by this
        self.width_division_factor = width_division_factor  # Width constraint (e.g., 16). Final width must be divisible by this

    def crop_and_resize(self, image, target_height, target_width):
        """
        Resize image to cover target size, then center crop.

        ALGORITHM:
        1. Calculate scale to cover target (max of width_scale, height_scale)
        2. Resize image with that scale
        3. Center crop to exact target size

        This ensures no black bars, no distortion, output is exactly target size.

        Args:
            image: PIL.Image
            target_height: Desired height
            target_width: Desired width

        Returns:
            Resized and cropped PIL.Image
        """
        width, height = image.size  # Get original dimensions. PIL uses (width, height) order
        scale = max(target_width / width, target_height / height)  # Calculate scale to COVER target. max() ensures both dimensions meet or exceed target (no black bars). Example: 1920×1080 → 1024×1024, scale = max(1024/1920, 1024/1080) = max(0.53, 0.95) = 0.95
        image = torchvision.transforms.functional.resize(
            image,  # Input image
            (round(height*scale), round(width*scale)),  # New size after scaling. Maintains aspect ratio. Example: 1080*0.95=1026, 1920*0.95=1824 → (1026, 1824)
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR  # Interpolation method. BILINEAR is good balance of quality/speed
        )  # Resize image to cover target (at least one dimension will exceed target)
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))  # Crop from center to exact target size. Removes excess pixels. Example: (1026, 1824) → (1024, 1024) by removing 1 pixel top+bottom, 400 pixels left+right
        return image  # Return image of exact size (target_height, target_width) with no distortion

    def get_height_width(self, image):
        """
        Calculate target height and width for this image.

        TWO MODES:
        1. Fixed size (self.height and self.width provided):
           - Return those dimensions directly

        2. Dynamic size (self.height or self.width is None):
           - Start with original dimensions
           - Scale down if exceeds max_pixels
           - Round to nearest division_factor multiple

        Args:
            image: PIL.Image

        Returns:
            (height, width) tuple
        """
        if self.height is None or self.width is None:  # DYNAMIC SIZE MODE (at least one dimension is None)
            width, height = image.size  # Get original image dimensions. PIL returns (width, height)
            if width * height > self.max_pixels:  # If image too large (exceeds max_pixels limit)
                scale = (width * height / self.max_pixels) ** 0.5  # Calculate scale to fit in max_pixels. scale = sqrt(current_pixels / max_pixels). Example: 1920×1080=2073600, max=1048576, scale=sqrt(2073600/1048576)=1.41
                height, width = int(height / scale), int(width / scale)  # Scale down dimensions. Example: 1080/1.41=765, 1920/1.41=1361 → (765, 1361)
            height = height // self.height_division_factor * self.height_division_factor  # Round down to nearest multiple of division factor. Example: 765//16=47, 47*16=752
            width = width // self.width_division_factor * self.width_division_factor  # Round down to nearest multiple of division factor. Example: 1361//16=85, 85*16=1360
        else:  # FIXED SIZE MODE (both dimensions provided)
            height, width = self.height, self.width  # Use provided fixed dimensions
        return height, width  # Return target dimensions (guaranteed to be divisible by division factors)

    def __call__(self, data: Image.Image):
        """
        Apply crop and resize to image.

        Args:
            data: PIL.Image

        Returns:
            Resized and cropped PIL.Image
        """
        image = self.crop_and_resize(data, *self.get_height_width(data))  # Calculate target size, then crop and resize. *unpacks (height, width) as arguments
        return image  # Return processed image with correct dimensions



# ============================================================================
# UTILITY OPERATORS
# ============================================================================

class ToList(DataProcessingOperator):
    """
    Wrap data in a list.

    Useful for converting single images to video format (list of frames).

    Example: PIL.Image → [PIL.Image]
    """

    def __call__(self, data):
        """Wrap data in list."""
        return [data]  # Convert single item to list with one element. Used to treat image as 1-frame video


# ============================================================================
# VIDEO LOADING OPERATORS
# ============================================================================
# These operators handle video files and GIFs:
# - LoadVideo: Load video frames using imageio (MP4, AVI, MOV, etc.)
# - LoadGIF: Load GIF frames using imageio.v3
# Both support:
#   - Frame count constraints (time_division_factor, time_division_remainder)
#   - Per-frame preprocessing (frame_processor)
# ============================================================================

class LoadVideo(DataProcessingOperator):
    """
    Load video frames from file using imageio.

    Supports: MP4, AVI, MOV, WMV, MKV, FLV, WEBM

    FEATURES:
    - Respects frame count constraints (e.g., num_frames % 4 == 1)
    - Applies frame_processor to each frame (for efficient preprocessing)
    - Returns list of PIL.Image frames

    EXAMPLE:
    LoadVideo(
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=ImageCropAndResize(512, 512, ...)
    )
    This loads 81 frames (or fewer if video is short), where 81 % 4 == 1
    """

    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        """
        Initialize video loader.

        Args:
            num_frames: Target number of frames to load (default: 81)
            time_division_factor: Frame count constraint divisor (default: 4)
            time_division_remainder: Required remainder (default: 1)
                → (num_frames % time_division_factor == time_division_remainder)
            frame_processor: Function to apply to each frame (default: identity)
                → Common: ImageCropAndResize for resizing during load
        """
        self.num_frames = num_frames  # Target number of frames (e.g., 81). May be reduced if video is shorter
        self.time_division_factor = time_division_factor  # Temporal constraint divisor (e.g., 4). num_frames must satisfy modulo condition
        self.time_division_remainder = time_division_remainder  # Required remainder (e.g., 1). num_frames % 4 == 1 means valid counts: 81, 85, 89, etc.
        # frame_processor is built in the video loader for high efficiency.
        self.frame_processor = frame_processor  # Function to preprocess each frame. Applied during loading for efficiency. Example: resize, crop, normalize

    def get_num_frames(self, reader):
        """
        Calculate actual number of frames to load.

        ALGORITHM:
        1. Start with target num_frames
        2. If video has fewer frames, use video's frame count
        3. Reduce until frame count satisfies modulo constraint

        Args:
            reader: imageio video reader

        Returns:
            Number of frames to load (satisfies time constraints)

        EXAMPLE:
        - Target: 81 frames
        - Video has: 60 frames
        - 60 % 4 = 0 (not 1), so reduce: 59, 58, 57 (57 % 4 = 1 ✓)
        - Return: 57
        """
        num_frames = self.num_frames  # Start with target frame count (e.g., 81)
        if int(reader.count_frames()) < num_frames:  # If video has fewer frames than target
            num_frames = int(reader.count_frames())  # Use video's actual frame count
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:  # While frame count doesn't satisfy modulo constraint. Loop: reduce num_frames by 1 until constraint satisfied
                num_frames -= 1  # Reduce by 1 and check again. Example: 60→59→58→57 (57%4=1✓)
        return num_frames  # Return valid frame count (satisfies modulo constraint)

    def __call__(self, data: str):
        """
        Load video frames from file path.

        Args:
            data: Path to video file (MP4, AVI, MOV, etc.)

        Returns:
            List of PIL.Image frames (preprocessed with frame_processor)
        """
        reader = imageio.get_reader(data)  # Open video file with imageio. Returns reader object for frame-by-frame access
        num_frames = self.get_num_frames(reader)  # Calculate how many frames to load (respects constraints)
        frames = []  # Initialize list to store frames
        for frame_id in range(num_frames):  # Load frames 0 to num_frames-1
            frame = reader.get_data(frame_id)  # Read frame as numpy array (H, W, C). Values in [0, 255]
            frame = Image.fromarray(frame)  # Convert numpy array to PIL.Image
            frame = self.frame_processor(frame)  # Apply preprocessing (e.g., resize, crop). Happens during load for efficiency
            frames.append(frame)  # Add processed frame to list
        reader.close()  # Close video file to free resources
        return frames  # Return list of PIL.Image frames



class SequencialProcess(DataProcessingOperator):
    """
    Apply an operator to each element in a sequence.

    Useful for processing lists of items with the same operator.

    EXAMPLE:
    SequencialProcess(ImageCropAndResize(512, 512, ...))
    Applied to [img1, img2, img3] → [resize(img1), resize(img2), resize(img3)]
    """

    def __init__(self, operator=lambda x: x):
        """
        Initialize sequential processor.

        Args:
            operator: Function to apply to each element (default: identity)
        """
        self.operator = operator  # Store operator to apply. Can be any callable (function, class instance with __call__, etc.)

    def __call__(self, data):
        """
        Apply operator to each element in data.

        Args:
            data: Iterable (list, tuple, etc.)

        Returns:
            List of processed elements
        """
        return [self.operator(i) for i in data]  # List comprehension: apply operator to each element. Example: [resize(img) for img in images]


class LoadGIF(DataProcessingOperator):
    """
    Load GIF frames using imageio.v3.

    Similar to LoadVideo but specialized for GIF format.
    Uses imageio.v3.imread which loads all frames at once.

    FEATURES:
    - Respects frame count constraints (e.g., num_frames % 4 == 1)
    - Applies frame_processor to each frame
    - Returns list of PIL.Image frames

    EXAMPLE:
    LoadGIF(
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=ImageCropAndResize(512, 512, ...)
    )
    """

    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        """
        Initialize GIF loader.

        Args:
            num_frames: Target number of frames to load (default: 81)
            time_division_factor: Frame count constraint divisor (default: 4)
            time_division_remainder: Required remainder (default: 1)
            frame_processor: Function to apply to each frame (default: identity)
        """
        self.num_frames = num_frames  # Target number of frames (e.g., 81). May be reduced if GIF has fewer frames
        self.time_division_factor = time_division_factor  # Temporal constraint divisor (e.g., 4). num_frames must satisfy modulo condition
        self.time_division_remainder = time_division_remainder  # Required remainder (e.g., 1). num_frames % 4 == 1
        # frame_processor is built in the video loader for high efficiency.
        self.frame_processor = frame_processor  # Function to preprocess each frame during loading

    def get_num_frames(self, path):
        """
        Calculate actual number of frames to load from GIF.

        ALGORITHM:
        1. Load all frames to get count
        2. Start with target num_frames
        3. If GIF has fewer frames, use GIF's frame count
        4. Reduce until frame count satisfies modulo constraint

        Args:
            path: Path to GIF file

        Returns:
            Number of frames to load (satisfies time constraints)
        """
        num_frames = self.num_frames  # Start with target frame count (e.g., 81)
        images = iio.imread(path, mode="RGB")  # Load entire GIF to get frame count. mode="RGB" ensures RGB format
        if len(images) < num_frames:  # If GIF has fewer frames than target
            num_frames = len(images)  # Use GIF's actual frame count
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:  # While frame count doesn't satisfy modulo constraint
                num_frames -= 1  # Reduce by 1 and check again
        return num_frames  # Return valid frame count

    def __call__(self, data: str):
        """
        Load GIF frames from file path.

        Args:
            data: Path to GIF file

        Returns:
            List of PIL.Image frames (preprocessed with frame_processor)
        """
        num_frames = self.get_num_frames(data)  # Calculate how many frames to load (respects constraints)
        frames = []  # Initialize list to store frames
        images = iio.imread(data, mode="RGB")  # Load all GIF frames as numpy arrays. Returns list of (H, W, 3) arrays
        for img in images:  # Iterate through each frame (numpy array)
            frame = Image.fromarray(img)  # Convert numpy array to PIL.Image
            frame = self.frame_processor(frame)  # Apply preprocessing (e.g., resize, crop)
            frames.append(frame)  # Add processed frame to list
            if len(frames) >= num_frames:  # If we've loaded enough frames
                break  # Stop loading (don't process remaining frames)
        return frames  # Return list of PIL.Image frames
    


# ============================================================================
# ROUTING OPERATORS - Dynamic Operator Selection
# ============================================================================
# These operators choose which operator to apply based on:
# - RouteByExtensionName: File extension (jpg vs mp4 vs gif)
# - RouteByType: Data type (str vs list vs PIL.Image)
#
# Enables flexible pipelines that handle multiple input types
# ============================================================================

class RouteByExtensionName(DataProcessingOperator):
    """
    Route data to different operators based on file extension.

    Useful for handling mixed datasets (images and videos).

    WHAT IS operator_map?
    operator_map is a list of (condition, operator) pairs - like a switch statement.
    The router checks each pair in order and uses the FIRST MATCH.

    EXAMPLE:
    operator_map = [
        (("jpg", "png"), LoadImage()),           # If extension is jpg/png
        (("mp4", "avi"), LoadVideo()),           # If extension is mp4/avi
        (("gif",), LoadGIF()),                   # If extension is gif
    ]

    router = RouteByExtensionName(operator_map)

    # Usage:
    router("photo.jpg")   # Matches first pair → LoadImage()("photo.jpg")
    router("clip.mp4")    # Matches second pair → LoadVideo()("clip.mp4")
    router("anim.gif")    # Matches third pair → LoadGIF()("anim.gif")

    HOW ROUTING WORKS:
    for extensions, operator in operator_map:
        if file_extension in extensions:
            return operator(data)  # First match wins!
    """

    def __init__(self, operator_map):
        """
        Initialize router with extension→operator mapping.

        Args:
            operator_map: List of (extensions, operator) tuples
                extensions: Tuple of file extensions (e.g., ("jpg", "png"))
                    or None for catch-all
                operator: DataProcessingOperator to use for these extensions

        EXAMPLE operator_map:
        [
            (("jpg", "png", "webp"), LoadImage()),     # Images → LoadImage
            (("mp4", "avi", "mov"), LoadVideo()),      # Videos → LoadVideo
            (("gif",), LoadGIF()),                      # GIFs → LoadGIF
            (None, DefaultOperator()),                  # Anything else → DefaultOperator
        ]
        """
        self.operator_map = operator_map  # Store list of (extensions, operator) pairs. Checked in order, first match is used

    def __call__(self, data: str):
        """
        Route file path to appropriate operator based on extension.

        Args:
            data: File path (string)

        Returns:
            Result of applying matched operator to data
        """
        file_ext_name = data.split(".")[-1].lower()  # Extract file extension. Split by ".", take last part, convert to lowercase. Example: "video.MP4" → "mp4"
        for ext_names, operator in self.operator_map:  # Iterate through (extensions, operator) pairs in order
            if ext_names is None or file_ext_name in ext_names:  # If this is catch-all (None) or extension matches
                return operator(data)  # Apply this operator and return result. First match wins
        raise ValueError(f"Unsupported file: {data}")  # No operator matched this extension, raise error


class RouteByType(DataProcessingOperator):
    """
    Route data to different operators based on Python type.

    Useful for handling data that can be str (path) or list (paths) or PIL.Image.

    WHAT IS operator_map?
    operator_map is a list of (type_condition, operator) pairs - like a switch statement on data type.
    The router checks each pair in order and uses the FIRST MATCH.

    EXAMPLE:
    operator_map = [
        (str, LoadImage()),                      # If input is string (file path)
        (list, SequencialProcess(LoadImage())),  # If input is list (multiple paths)
        (Image.Image, lambda x: x),              # If input is already PIL.Image
    ]

    router = RouteByType(operator_map)

    # Usage:
    router("photo.jpg")                    # Matches first → LoadImage()("photo.jpg")
    router(["a.jpg", "b.jpg"])             # Matches second → SequencialProcess(...)
    router(some_pil_image)                 # Matches third → return as-is

    HOW ROUTING WORKS:
    for dtype, operator in operator_map:
        if isinstance(data, dtype):
            return operator(data)  # First match wins!
    """

    def __init__(self, operator_map):
        """
        Initialize router with type→operator mapping.

        Args:
            operator_map: List of (dtype, operator) tuples
                dtype: Python type (str, list, Image.Image, etc.) or None for catch-all
                operator: DataProcessingOperator to use for this type

        EXAMPLE operator_map:
        [
            (str, ToAbsolutePath(...) >> LoadImage()),        # String path → load image
            (list, SequencialProcess(LoadImage())),           # List of paths → load all
            (Image.Image, lambda x: x),                       # Already loaded → pass through
            (None, DefaultOperator()),                        # Anything else → default
        ]
        """
        self.operator_map = operator_map  # Store list of (type, operator) pairs. Checked in order, first match is used

    def __call__(self, data):
        """
        Route data to appropriate operator based on type.

        Args:
            data: Input data (any type)

        Returns:
            Result of applying matched operator to data
        """
        for dtype, operator in self.operator_map:  # Iterate through (type, operator) pairs in order
            if dtype is None or isinstance(data, dtype):  # If this is catch-all (None) or data matches type
                return operator(data)  # Apply this operator and return result. First match wins
        raise ValueError(f"Unsupported data: {data}")  # No operator matched this type, raise error


# ============================================================================
# MISCELLANEOUS OPERATORS
# ============================================================================

class LoadTorchPickle(DataProcessingOperator):
    """
    Load PyTorch pickle file (.pth).

    Used for loading cached/preprocessed data.

    EXAMPLE:
    LoadTorchPickle(map_location="cpu")
    Loads .pth file to CPU (useful for distributed training)
    """

    def __init__(self, map_location="cpu"):
        """
        Initialize pickle loader.

        Args:
            map_location: Device to load tensors to (e.g., "cpu", "cuda")
        """
        self.map_location = map_location  # Device to load tensors to. "cpu" avoids CUDA errors if file saved on different GPU

    def __call__(self, data):
        """
        Load pickle file.

        Args:
            data: Path to .pth file

        Returns:
            Loaded object (usually dict with tensors)
        """
        return torch.load(data, map_location=self.map_location, weights_only=False)  # Load pickle file. map_location moves tensors to specified device. weights_only=False allows loading arbitrary Python objects (not just tensors)


class ToAbsolutePath(DataProcessingOperator):
    """
    Convert relative path to absolute path.

    Prepends base_path to relative paths in metadata.

    EXAMPLE:
    ToAbsolutePath(base_path="/data/videos")
    "clip1.mp4" → "/data/videos/clip1.mp4"
    """

    def __init__(self, base_path=""):
        """
        Initialize path converter.

        Args:
            base_path: Base directory to prepend (default: "")
        """
        self.base_path = base_path  # Store base path to prepend. Usually the dataset root directory

    def __call__(self, data):
        """
        Convert relative path to absolute path.

        Args:
            data: Relative path (string)

        Returns:
            Absolute path (string)
        """
        return os.path.join(self.base_path, data)  # Join base path with relative path. Example: os.path.join("/data", "video.mp4") → "/data/video.mp4"



# ============================================================================
# CLASS: UnifiedDataset - Main Dataset Class for Training
# ============================================================================
# PURPOSE: Flexible PyTorch Dataset for loading images/videos with metadata
#
# TWO LOADING MODES:
# 1. Metadata mode (metadata_path provided):
#    - Loads CSV/JSON/JSONL with metadata
#    - Each row has paths to data files (video, image, audio, etc.)
#    - Operators load and preprocess files on-the-fly
#
# 2. Cached mode (metadata_path is None):
#    - Searches for .pth files in base_path
#    - Loads preprocessed data from pickle files
#    - Faster but requires pre-caching
#
# KEY CONCEPTS:
# - data_file_keys: Which metadata columns contain file paths
# - main_data_operator: Default operator for file loading
# - special_operator_map: Custom operators for specific keys
#
# EXAMPLE USAGE (S2V training):
# dataset = UnifiedDataset(
#     base_path="/data/RenderMe360",
#     metadata_path="/data/RenderMe360/metadata.csv",
#     data_file_keys=("video", "audio"),
#     main_data_operator=default_video_operator(...),
#     special_operator_map={
#         "audio": ToAbsolutePath(...) >> LoadAudio()
#     },
# )
# ============================================================================

class UnifiedDataset(torch.utils.data.Dataset):
    """
    Flexible dataset for loading images/videos with metadata.

    Supports two modes:
    1. Metadata mode: Load from CSV/JSON with file paths
    2. Cached mode: Load preprocessed .pth files

    Used by all training scripts in DiffSynth-Studio.
    """

    def __init__(
        self,
        base_path=None,              # Base directory for data files
        metadata_path=None,           # Path to metadata file (CSV/JSON/JSONL)
        repeat=1,                     # Repeat dataset N times per epoch
        data_file_keys=tuple(),       # Keys that contain file paths
        main_data_operator=lambda x: x,  # Default operator for loading files
        special_operator_map=None,    # Custom operators for specific keys
    ):
        """
        Initialize UnifiedDataset.

        Args:
            base_path: Root directory containing data files
            metadata_path: Path to metadata file (CSV/JSON/JSONL) or None for cached mode
            repeat: Dataset repetition factor (for small datasets)
            data_file_keys: Tuple of keys that contain file paths (e.g., ("video", "image"))
            main_data_operator: Default operator to load files
            special_operator_map: Dict mapping keys to custom operators
                Example: {"audio": LoadAudio(), "mask": LoadMask()}

        TWO MODES:
        - If metadata_path provided: Load from metadata (on-the-fly loading)
        - If metadata_path is None: Load from cached .pth files (fast loading)
        """
        self.base_path = base_path  # Store base directory. All relative paths in metadata are relative to this
        self.metadata_path = metadata_path  # Store metadata path. None means use cached mode
        self.repeat = repeat  # Dataset size multiplier. Useful for small datasets (e.g., 100 samples → 1000 with repeat=10)
        self.data_file_keys = data_file_keys  # Tuple of keys to load as files. Example: ("video", "audio", "image")
        self.main_data_operator = main_data_operator  # Default operator for loading files. Applied to keys in data_file_keys (unless overridden in special_operator_map)
        self.cached_data_operator = LoadTorchPickle()  # Operator for loading cached .pth files (used in cached mode)
        self.special_operator_map = {} if special_operator_map is None else special_operator_map  # Custom operators for specific keys. Overrides main_data_operator. Example: {"audio": LoadAudio()} means "audio" key uses LoadAudio() instead of main_data_operator
        self.data = []  # List of metadata dicts (each dict is one sample). Populated in load_metadata()
        self.cached_data = []  # List of cached .pth file paths. Populated if metadata_path is None
        self.load_from_cache = metadata_path is None  # Flag: True = cached mode, False = metadata mode
        self.load_metadata(metadata_path)  # Load metadata or search for cached files
    
    # ========================================================================
    # ⭐ HELPER METHODS: Pre-built Operator Pipelines
    # ========================================================================
    # These static methods create common operator pipelines for convenience
    # ========================================================================

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        """
        Create default operator pipeline for loading images.

        Handles two input types:
        - str: Single image path → PIL.Image
        - list: Multiple image paths → [PIL.Image, ...]

        Pipeline: path → absolute path → load image → crop & resize

        Args:
            base_path: Base directory for relative paths
            max_pixels: Maximum resolution (dynamic mode)
            height, width: Fixed size (None = dynamic)
            height_division_factor, width_division_factor: Dimension constraints

        Returns:
            RouteByType operator that handles str or list inputs
        """
        return RouteByType(operator_map=[
            # If input is string (single image path)
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),  # Chain: "img.jpg" → "/data/img.jpg" → PIL.Image → resized PIL.Image
            # If input is list (multiple image paths)
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),  # Apply same pipeline to each image in list
        ])

    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        """
        Create default operator pipeline for loading videos.

        Handles various video formats and single images:
        - Image files (jpg, png, etc.) → [PIL.Image] (1-frame video)
        - GIF files → [PIL.Image, ...] (multi-frame)
        - Video files (mp4, avi, etc.) → [PIL.Image, ...] (multi-frame)

        Pipeline: path → absolute path → detect format → load → crop & resize frames

        Args:
            base_path: Base directory for relative paths
            max_pixels: Maximum resolution per frame
            height, width: Fixed frame size (None = dynamic)
            height_division_factor, width_division_factor: Frame dimension constraints
            num_frames: Target frame count
            time_division_factor, time_division_remainder: Frame count constraints

        Returns:
            RouteByType operator that handles different video formats

        EXAMPLE:
        operator = default_video_operator(num_frames=81, ...)
        - "video.mp4" → [PIL.Image×81]
        - "image.jpg" → [PIL.Image×1]
        - "anim.gif" → [PIL.Image×N] where N satisfies constraints
        """
        return RouteByType(operator_map=[
            # If input is string (file path), route by extension
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                # If image file (jpg, png, etc.)
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),  # Load image, resize, wrap in list (1-frame video). Example: "img.jpg" → PIL.Image → resized PIL.Image → [resized PIL.Image]
                # If GIF file
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,  # Frame count constraints
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),  # Resize each frame during load (efficient)
                )),  # Load GIF frames with preprocessing. Example: "anim.gif" → [PIL.Image, PIL.Image, ...] (resized during load)
                # If video file (mp4, avi, mov, etc.)
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,  # Frame count constraints
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),  # Resize each frame during load (efficient)
                )),  # Load video frames with preprocessing. Example: "video.mp4" → [PIL.Image×81] (resized during load)
            ])),
        ])
        
    # ========================================================================
    # ⭐ METADATA LOADING METHODS
    # ========================================================================

    def search_for_cached_data_files(self, path):
        """
        Recursively search directory for .pth files (cached data).

        Used in cached mode (metadata_path is None).

        Args:
            path: Directory to search
        """
        for file_name in os.listdir(path):  # Iterate through files/folders in directory
            subpath = os.path.join(path, file_name)  # Create full path
            if os.path.isdir(subpath):  # If this is a subdirectory
                self.search_for_cached_data_files(subpath)  # Recursively search subdirectory. Depth-first search through entire tree
            elif subpath.endswith(".pth"):  # If this is a .pth file (cached data)
                self.cached_data.append(subpath)  # Add to list of cached files. Each .pth file is one sample

    def load_metadata(self, metadata_path):
        """
        Load metadata from file or search for cached data.

        THREE METADATA FORMATS SUPPORTED:
        1. CSV: Pandas-readable (most common)
        2. JSON: Single array of dicts
        3. JSONL: One JSON dict per line

        Args:
            metadata_path: Path to metadata file or None for cached mode
        """
        if metadata_path is None:  # CACHED MODE (no metadata file)
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)  # Recursively find all .pth files in base_path
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):  # JSON FORMAT: [{"video": "...", "audio": "..."}, ...]
            with open(metadata_path, "r") as f:  # Open JSON file
                metadata = json.load(f)  # Load entire JSON as list of dicts
            self.data = metadata  # Store list of sample dicts
        elif metadata_path.endswith(".jsonl"):  # JSONL FORMAT: one JSON dict per line
            metadata = []  # Initialize list to store samples
            with open(metadata_path, 'r') as f:  # Open JSONL file
                for line in f:  # Read line by line
                    metadata.append(json.loads(line.strip()))  # Parse each line as JSON dict, add to list
            self.data = metadata  # Store list of sample dicts
        else:  # CSV FORMAT (default): pandas readable
            metadata = pandas.read_csv(metadata_path)  # Load CSV using pandas. Returns DataFrame
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]  # Convert each row to dict. iloc[i] gets row i as Series, to_dict() converts to dict

    # ========================================================================
    # ⭐ PYTORCH DATASET INTERFACE: __getitem__ and __len__
    # ========================================================================

    def __getitem__(self, data_id):
        """
        Get one sample from dataset.

        TWO LOADING MODES:
        1. Cached mode: Load preprocessed .pth file
        2. Metadata mode: Load files on-the-fly using operators

        Args:
            data_id: Sample index (0 to len(dataset)-1)

        Returns:
            Dict with loaded data (keys depend on metadata/cached data)

        METADATA MODE EXAMPLE:
        data_id=0 → self.data[0] = {"video": "clip1.mp4", "audio": "clip1.wav"}
        → Apply operators → {"video": [PIL.Image×81], "audio": torch.Tensor}
        """
        if self.load_from_cache:  # CACHED MODE
            data = self.cached_data[data_id % len(self.cached_data)]  # Get path to cached file. % allows repeat to work (indices wrap around)
            data = self.cached_data_operator(data)  # Load .pth file using LoadTorchPickle(). Returns dict with preprocessed data
        else:  # METADATA MODE
            data = self.data[data_id % len(self.data)].copy()  # Get metadata dict for this sample. % allows repeat to work. .copy() prevents modifying original metadata
            for key in self.data_file_keys:  # Iterate through keys that should be loaded as files (e.g., "video", "audio")
                if key in data:  # If this key exists in metadata
                    if key in self.special_operator_map:  # If special operator defined for this key
                        data[key] = self.special_operator_map[key](data[key])  # Apply special operator. Example: special_operator_map["audio"](data["audio"]) loads audio file
                    elif key in self.data_file_keys:  # If no special operator (redundant check, always True here)
                        data[key] = self.main_data_operator(data[key])  # Apply main operator. Example: main_data_operator(data["video"]) loads video file
        return data  # Return dict with loaded data (PIL.Images, tensors, etc.)

    def __len__(self):
        """
        Get dataset size.

        Size = (number of samples) × repeat

        Returns:
            Total dataset size
        """
        if self.load_from_cache:  # Cached mode
            return len(self.cached_data) * self.repeat  # Number of .pth files × repeat factor
        else:  # Metadata mode
            return len(self.data) * self.repeat  # Number of metadata rows × repeat factor

    def check_data_equal(self, data1, data2):
        """
        Check if two data dicts are equal (debug utility).

        Args:
            data1: First dict
            data2: Second dict

        Returns:
            True if equal, False otherwise
        """
        # Debug only
        if len(data1) != len(data2):  # If different number of keys
            return False  # Not equal
        for k in data1:  # Check each key in data1
            if data1[k] != data2[k]:  # If values differ
                return False  # Not equal
        return True  # All keys and values match


# ============================================================================
# KEY TAKEAWAYS: Understanding the Data Loading System
# ============================================================================
"""
✅ WHAT YOU LEARNED FROM THIS FILE:

1. **DataProcessingPipeline** - Chain operators together:
   - Use >> operator to chain: LoadImage() >> Resize(512, 512)
   - Operators are applied sequentially
   - Composable and reusable

2. **DataProcessingOperator** - Base class for transformations:
   - LoadImage: PIL.Image loading
   - LoadVideo: Video frame loading (respects time constraints)
   - LoadGIF: GIF frame loading
   - ImageCropAndResize: Smart resizing (aspect ratio + constraints)
   - RouteByExtensionName/Type: Dynamic operator selection
   - ToAbsolutePath: Convert relative paths to absolute

3. **UnifiedDataset** - Main dataset class:
   - TWO MODES:
     * Metadata mode: Load from CSV/JSON with file paths
     * Cached mode: Load preprocessed .pth files
   - data_file_keys: Which metadata columns contain file paths
   - main_data_operator: Default operator for loading
   - special_operator_map: Custom operators for specific keys
   - __getitem__() applies operators to load data on-the-fly

4. **Operator Chaining Flow** (S2V example):
   Metadata: {"video": "clip1.mp4", "audio": "clip1.wav"}
   ↓
   data_file_keys = ("video", "audio")
   ↓
   main_data_operator = default_video_operator(...)
   ↓
   "clip1.mp4" → ToAbsolutePath → RouteByExtensionName → LoadVideo → [PIL.Image×81]
   "clip1.wav" → special_operator_map["audio"] → torch.Tensor
   ↓
   Result: {"video": [PIL.Image×81], "audio": torch.Tensor}

📊 HOW THIS IS USED IN S2V TRAINING:
1. Training script creates UnifiedDataset with RenderMe360 metadata
2. Each __getitem__() loads video frames and audio
3. Training module preprocesses (PIL → torch.Tensor, VAE encode)
4. DiT receives latents + audio embeddings
5. Training loss computed and backpropagated

🎯 NEXT STEPS:
Now you understand data loading! Next, study:
- Training script: How dataset is used in training loop
- WanVideoUnit_S2V: How audio is processed for conditioning
- Training module: How data flows from dataset to model
"""
