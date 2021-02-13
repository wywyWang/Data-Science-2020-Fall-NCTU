# Data-Science-2020-Fall-NCTU

## HW1
### 1-1 Crawl [PTT Beauty](https://www.ptt.cc/bbs/Beauty/index.html)
- Implemented functions
  - `crawl`
    
    Crawl all posts in 2019.
    ```=python
    python {student_id}.py crawl
    ```
  - `push`
  
    Count all pushs and boos from all posts between start data and end data.
    ```=python
    python {student_id}.py push {start_date} {end_date}
    ```
  - `popular`
  
    Count # of pushs >= 100 and crawl all urls of photos including from pushs in the posts.
    ```=python
    python {student_id}.py popular {start_date} {end_date}
    ```
  - `keyword`
  
    Crawl all urls of photos including from pushs in the posts if there exists keyword in the context.
    ```=python
    python {student_id}.py popular {keyword} {start_date} {end_date}
    ```
### 1-2 Popularity Classification
- Input
  - An image from crawled data
- Output
  - Whether the image is in the popular article
- Evaluation metric: F1 score
## HW2 Headline Attractiveness Predictor
### 2-1 Attractiveness Annotation
Given a headline, annotate the corresponding attractive score from 1 to 5.
### 2-2 Attractiveness Prediction on [Kaggle](https://www.kaggle.com/c/datascience2020hw2)
Predict the attractiveness of a headline based on the content or the category.

Evaluation metric: mean square error (MSE)
## HW3 Panorama
Implement feature detection, feature matching, image matching, and blending.

Evaluation metric: Use pretrained VGG16 to get feature vector from groundtruth and generated images then compute cosine similarity.
## HW4 GAN
Train a GAN from scratch to generate dog images from Gaussian noise

Evaluation metric: Frechet inception distance (FID)
