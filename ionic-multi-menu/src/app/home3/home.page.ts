import {Component, OnInit} from '@angular/core';
import {FileUploadService} from "../file-upload.service";
import {basename} from "@angular-devkit/core";

@Component({
    selector: 'app-home',
    templateUrl: 'home.page.html',
    styleUrls: ['home.page.scss'],
})
export class HomePage implements OnInit{
    list = [];
    list2= [];

    listBoolean = [];

    img1 = 'http://localhost:5000/search?name=1.jpg'
    img2 = 'http://localhost:5000/search?name=2.jpg'
    img3 = 'http://localhost:5000/search?name=3.jpg'
    img4 = 'http://localhost:5000/search?name=4.jpg'

    numList = 9;
    row:any;
    col:any;

    currNum = 0;

    active: true;

    constructor(private uploadService: FileUploadService) {

    }

    ngOnInit(): void {
        this.uploadService.getNumber().subscribe(res=>{
            // console.log(res.result);
            let baseUrl = 'http://localhost:5000/search?name=';
            let baseUrl2 = 'http://localhost:5000/search2?name=';

            for (let i=0; i<res.result.length; i++){
                // console.log(res.result[i]);
                let url = baseUrl + res.result[i];
                let url2 = baseUrl2 + res.result[i];

                let all = [url,url2,true];

                this.list.push(url);
                this.list2.push(url2);
                this.listBoolean.push(true);
            }

            console.log('This is list inside upload:');
            console.log(this.list);
            console.log('This is length list inside upload:');
            console.log(this.list.length);

            this.numList = this.list.length;

        });

        console.log('This is list: ');
        console.log(this.list[0]);
        console.log(this.list[1]);
        console.log(this.list[2]);
        console.log(this.list[3]);

        console.log('This is num list: ');
        // this.numList = this.list.length;
        console.log(this.numList);

        this.col = 3;
        this.row = Math.floor(this.numList / this.col);

        console.log('Col: ');
        console.log(this.col);
        console.log('Row: ');
        console.log(this.row);
    }

    getImg1() {
        return this.img1 + '&' + (new Date()).getTime();
    }

    getImg(x) {
        console.log('This is getImg');

        // let num = Number(x);
        // if (this.list[num][2] === true){
        //     console.log('Get image num: ');
        //     let url = this.list[num][0] +'&' +  (new Date()).getTime();
        //     console.log(url);
        //     // x[2] = false;
        //
        //     return url;
        // }

        console.log('Get image num: ');
        let url = x +'&' +  (new Date()).getTime();
        console.log(url);

        return url;
    }

    getImg2() {
        return this.img2 + '&' + (new Date()).getTime();
    }

    getImg3() {
        return this.img3 + '&' + (new Date()).getTime();
    }

    getImg4() {
        return this.img4 + '&' + (new Date()).getTime();
    }

    getNumRow() {
        let index = 0;
        let res = [];

        console.log('This row:');
        console.log(this.row);

        for (let i=0; i<this.row; i++){
            res.push(index);
            index+=1;
        }

        console.log('Get num row');
        console.log(res);

        return res;
    }

    getListCol(x:number) {
        console.log('Current number of x: ');
        console.log(x);

        let curr = x*3;
        let res = [];

        for (let i=curr; i<curr+3; i++){
            res.push(this.list[i]);
        }

        console.log('Get list col');
        console.log(res);
        return res;
    }

    clickImg(j: any) {
        console.log('This is click ');
        console.log(j);
        this.uploadService.swapStyle(j).subscribe(res=>{
            console.log('Swap Style');
        }).unsubscribe();
    }
}
