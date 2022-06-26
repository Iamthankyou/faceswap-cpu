import {Component} from '@angular/core';
import {FileUploadService} from "../file-upload.service";

@Component({
    selector: 'app-home',
    templateUrl: 'home.page.html',
    styleUrls: ['home.page.scss'],
})
export class HomePage {
    urlVideo = `http://localhost:4200/assets/images/demo/output.mp4`;
    timeStamp: any;

    constructor(private uploadService: FileUploadService) {
    }


    swapface() {

    }

    getVideo() {
        return this.urlVideo + '?' + (new Date()).getTime();
        // return this.urlVideo;
    }

}
