# DL_tutorials

### spacy en 离线安装

`
* pip install spacy -i https://pypi.tuna.tsinghua.edu.cn/simple

* cd en_core_web_sm-2.2.5

* python3 setup.py install

`

### 数据并行
<pre name="code", class="python">
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)
</pre>
