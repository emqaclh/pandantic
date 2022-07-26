# Overview

<p align="center">
    <img src="./docs/img/header_image.svg" width="348px">
</p>
<blockquote align="right">
  <q>To be an insufferable twit by arguing by definition</q>
</blockquote>

<div align="right">
    &ndash; <a href="https://www.urbandictionary.com/define.php?term=Pandantic" style="text-align: right"> Urban Dictionary</a>
</div>

[**pandantic**](/) is a complementary tool for [**pandas**](https://pandas.pydata.org/) to write and evaluate in a simple way the assumptions that your data must fulfill. Writing these assumptions is done in a simple way by declaring a data schema.

Each schema contains as many attributes as columns your data must contain and each column contains as many validations as your data must fulfill.

In addition, if a validation fails, it is possible to amend the data _on-the-fly_, thus keeping the validation flow simple and building a centralized way to declare the main modifications made to the data.

In this way, all your raw data is processed to build validated datasets that are homologous to each other.

[**pandantic**](/) is intended both to validate data in everyday analysis exercises and to process data in a **pipeline** in production.

Whatever the case, [**pandantic**](/) is a simple and useful tool for developers and data analysts who are comfortable with model-declaration based tools such as [**django**](https://www.djangoproject.com/) or [**pydantic**](https://pydantic-docs.helpmanual.io/).

The main advantage of [**pandantic**](/) is to allow to easily deal with incorrect validations by applying amendments, however, it is still a simple and recent tool. If a mature and more developed tool is required, we cannot but recommend [**pandera**](https://pandera.readthedocs.io/en/stable/).
