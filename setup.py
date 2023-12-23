from setuptools import setup

setup(
    name='yt2srt',
    version='0.1.0',
    py_modules=['yt2srt'],
    install_requires=[
        'Click',
        'pytube',
        'transformers',
        'torch'
    ],
    entry_points={
        'console_scripts': [
            'yt2srt = yt2srt:main',
        ],
    },
)