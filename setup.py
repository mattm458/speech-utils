import setuptools

setuptools.setup(
    name="speech_utils",
    version="0.1",
    packages=setuptools.find_packages(),
    package_data={"speech_utils": ["preprocessing/processing.praat"]},
)
