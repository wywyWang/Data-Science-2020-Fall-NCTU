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
- Evaluation metric

  F1 score
