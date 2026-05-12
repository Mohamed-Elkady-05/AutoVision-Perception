import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray


class FeatureExtractor:
    """
    Extract features from images.
    Current methods accept either file paths or numpy arrays representing images.
    Provides HOG and color histogram features to extract a concatenated feature vector.
    """

    def __init__(self, resize=None, hog_params=None, hist_bins=32):
        # resize: (width, height) or None
        self.resize = resize
        self.hist_bins = hist_bins
        self.hog_params = {} if hog_params is None else hog_params

    def _load_image(self, img):
        if isinstance(img, str):
            im = Image.open(img).convert('RGB')
            if self.resize:
                im = im.resize(self.resize)
            return np.asarray(im)
        elif hasattr(img, 'shape'):
            # numpy array must be either HxWxC or HxW
            im = img
            if self.resize:
                im = np.asarray(Image.fromarray(im).resize(self.resize))
            if im.ndim == 2:
                # grayscale to RGB
                im = np.stack([im, im, im], axis=-1)
            return im
        else:
            raise ValueError('Unsupported image input type')

    def extract_hog(self, img):
        im = self._load_image(img)
        gray = rgb2gray(im)
        features = hog(gray, **self.hog_params)
        return features

    def color_histogram(self, img, bins=None, normalize=True):
        im = self._load_image(img)
        bins = self.hist_bins if bins is None else bins
        chans = []
        for c in range(3):
            h, _ = np.histogram(im[:, :, c], bins=bins, range=(0, 255))
            chans.append(h.astype(np.float32))
        feat = np.concatenate(chans)
        if normalize:
            s = feat.sum()
            if s > 0:
                feat = feat / s
        return feat

    def extract(self, img, which=('hog', 'hist')):
        pieces = []
        if 'hog' in which:
            pieces.append(self.extract_hog(img))
        if 'hist' in which:
            pieces.append(self.color_histogram(img))
        if not pieces:
            raise ValueError('No feature types requested')
        return np.concatenate(pieces)

    def extract_from_list(self, images, which=('hog', 'hist'), verbose=False):
        """images: list of file paths or numpy arrays. Returns numpy array of features."""
        feats = []
        for i, im in enumerate(images):
            feats.append(self.extract(im, which=which))
            if verbose and (i + 1) % 100 == 0:
                print(f"Extracted features from {i+1} images")
        return np.vstack(feats)
