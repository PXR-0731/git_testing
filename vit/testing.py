import requests
import json


class Test_One():

    def setup_method(self):
        pass


    def test_vit(self):
        with open("./20.png", "rb") as f:
            file = {"file": ("./20.png", f, "image/jpeg")}
            res = requests.post("http://127.0.0.1:9009/vit", files=file) #, headers={"context-type":"application/json"}
            assert res.status_code==200
            print(json.loads(res.content))


    def teardown_method(self):
        pass



if __name__=='__main__':
    with open("20.png", "rb") as f:
        file = {"file": ("dog.jpg", f,"image/jpeg")}
        res = requests.post("http://127.0.0.1:9009/vit", files=file) #, headers={"context-type":"application/json"}

        print(json.loads(res.content))
        assert res.status_code==200