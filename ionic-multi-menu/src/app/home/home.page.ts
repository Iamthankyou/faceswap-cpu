import {ChangeDetectorRef, Component, OnInit} from '@angular/core';
import {FileUploadService} from "../file-upload.service";
import {Observable} from "rxjs";
import {DomSanitizer} from "@angular/platform-browser";

@Component({
    selector: 'app-home',
    templateUrl: 'home.page.html',
    styleUrls: ['home.page.scss'],
})
export class HomePage implements OnInit {

    imageInfos?: Observable<any>;

    imageToShow: any;

    image: any;

    const // @ts-ignore
    urlImage = `http://localhost:5000/files`;
    timeStamp: any;

    list = [
        {img: 'assets/images/1.jpg', name: 'Ionic5 Fruit App with Firebase'},
        {img: 'assets/images/2.jpg', name: 'Ionic 4 Online Clothes Shop App with Angular Admin Backend'},
        {img: 'assets/imagSes/3.jpg', name: 'Ionic 5 / Angular 8 Dark UI Theme / Template App | Starter App'},
        {img: 'assets/images/4.jpg', name: 'Ionic 5 / Angular 8 Gray UI Theme / Template App | Starter App'},
        {img: 'assets/images/5.jpg', name: 'Ionic 5 / Angular 8 UI Blue Theme / Template App | Starter App'},
        {img: 'assets/images/6.jpg', name: 'Ionic 5 / Angular 8 Red UI Theme / Template App | Starter App'}
    ];

    ngOnInit(): void {
        // this.swapface();
        this.urlImage = `http://localhost:5000/files`;
        this.timeStamp = (new Date()).getTime();
    }

    constructor(private uploadService: FileUploadService,
                private sanitizer: DomSanitizer,
                private cdRef: ChangeDetectorRef) {
    }

    getLinkPicture() {
        return this.urlImage + '?' + (new Date()).getTime();
    }

    createImageFromBlob(image: Blob) {
        let reader = new FileReader();
        reader.addEventListener("load", () => {
            this.imageToShow = reader.result;
        }, false);

        if (image) {
            reader.readAsDataURL(image);
        }

        console.log('Image to show');
        console.log(this.imageToShow);
    }

    swapface() {
        console.log("Swapface");
        let sub = this.uploadService.getFiles().subscribe(blob  => {
            this.createImageFromBlob(blob);
            // let objectURL = URL.createObjectURL(blob);
            // this.image = this.sanitizer.bypassSecurityTrustUrl(objectURL);
            // this.cdRef.detectChanges();
            // console.log(this.image);
            // this.imageToShow = res;
            // let objectURL = 'data:image/gif;base64' + blob.text();

            // this.imageToShow = this.sanitizer.bypassSecurityTrustUrl(objectURL);

            console.log(blob);
        });

    }
}
