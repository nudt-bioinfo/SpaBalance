from setuptools import Command, find_packages, setup

__lib_name__ = "SpaBalance"
__lib_version__ = "1.1.5"
__description__ = "Harmonizing Gradient Learning and Feature Disentanglement for Robust Spatial Multi-omics Integration"
__url__ = "https://github.com/nudt-bioinfo/SpaBalance"
__author__ = "Yong Zhao"
__author_email__ = "zhaoyong23a@nudt.edu.cn"
__license__ = "MIT"
__keywords__ = ["Spatial multi-omics", "Cross-omics integration", "Private and Shared Learning", " Multimodal Balanced Learning"]
__requires__ = ["requests",]

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ["SpaBalance"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = """Integrating multiple data modalities based on spatial information remains an unmet need in the analysis of spatial multi-omics data. Here, we propose **SpaBalance**, a spatial multi-omics deep integration model designed to decode spatial domains by leveraging graph neural networks (GNNs) and multilayer perceptrons (MLPs). SpaBalance first performs **intra-omics integration** by capturing within-modality relationships using spatial positioning and omics measurements, followed by **cross-omics integration** through a multi-head attention mechanism. To effectively learn shared features while preserving key modality-specific information, we employ a **dual-learning strategy** that combines modality-specific private learning with cross-modality shared learning. Additionally, to prevent any single modality from dominating the integration process, we introduce a **multi-modal balance learning approach**. We demonstrate that **SpaBalance** achieves more accurate spatial domain resolution across diverse tissue types and technological platforms, providing valuable biological insights into cross-modality spatial correlations. """,
    long_description_content_type="text/markdown"
)
