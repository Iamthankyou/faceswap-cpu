import { Component, OnInit } from '@angular/core';
import {Observable} from "rxjs";
import {FileUploadService} from "../file-upload.service";
import {HttpEventType, HttpResponse} from "@angular/common/http";

@Component({
  selector: 'app-style',
  templateUrl: './style.component.html',
  styleUrls: ['./style.component.scss'],
})
export class StyleComponent implements OnInit {

  selectedFiles1?: FileList;
  selectedFiles2?: FileList;

  progressInfos: any[] = [];
  message: string[] = [];

  previews1: string[] = [];
  previews2: string[] = [];

  video:any;
  videoRes:any;

  previews3: string[] = [];
  imageInfos?: Observable<any>;

  url;
  format;

  constructor(private uploadService: FileUploadService) { }

  ngOnInit(): void {
    this.url = './assets/images/dst.mp4';
    this.previews1.push(`http://localhost:5000/src?${Date.now()}`);
    this.previews2.push('./assets/images/dst.png');

    // this.videoRes = this.uploadService.getVideo().subscribe(responce => {
    //   // this.videoRes = JSON.parse(responce);
    //   console.log(responce);
    // });
  }

  onSelectFile(event) {
    const file = event.target.files && event.target.files[0];
    if (file) {
      console.log('Video File is exist');
      this.video = file;
      this.selectedFiles2 = file;

      var reader = new FileReader();
      reader.readAsDataURL(file);
      if(file.type.indexOf('image')> -1){
        this.format = 'image';
      } else if(file.type.indexOf('video')> -1){
        this.format = 'video';

      }
      reader.onload = (event) => {
        this.url = (<FileReader>event.target).result;
      }
    }
  }


  selectFiles1(event: any): void {
    this.message = [];
    this.progressInfos = [];
    this.selectedFiles1 = event.target.files;

    this.previews1 = [];
    if (this.selectedFiles1 && this.selectedFiles1[0]) {
      const numberOfFiles = this.selectedFiles1.length;
      for (let i = 0; i < numberOfFiles; i++) {
        const reader = new FileReader();

        reader.onload = (e: any) => {
          console.log(e.target.result);
          this.previews1.push(e.target.result);
        };

        reader.readAsDataURL(this.selectedFiles1[i]);
      }
    }
  }

  swapface() {
    console.log('Swapface');
    this.uploadService.getFiles().subscribe();
  }

  selectFiles2(event: any): void {
    this.message = [];
    this.progressInfos = [];
    this.selectedFiles2 = event.target.files;

    this.previews2 = [];
    if (this.selectedFiles2 && this.selectedFiles2[0]) {
      const numberOfFiles = this.selectedFiles2.length;
      for (let i = 0; i < numberOfFiles; i++) {
        const reader = new FileReader();

        reader.onload = (e: any) => {
          console.log(e.target.result);
          this.previews2.push(e.target.result);
        };

        reader.readAsDataURL(this.selectedFiles2[i]);
      }
    }
  }

  upload_face2(idx: number, file: File, name: string): void {
    this.progressInfos[idx] = { value: 0, fileName: name };

    if (file) {
      this.uploadService.upload_face2(file).subscribe({
        next: (event: any) => {
          if (event.type === HttpEventType.UploadProgress) {
            this.progressInfos[idx].value = Math.round(100 * event.loaded / event.total);
          } else if (event instanceof HttpResponse) {
            const msg = 'Uploaded the file successfully: ' + file.name;
            this.message.push(msg);
            this.imageInfos = this.uploadService.getFiles();
          }
        },
        error: (err: any) => {
          this.progressInfos[idx].value = 0;

          const msg = 'Could not upload the file: ' + file.name;
          this.message.push(msg);
        }});
    }
  }

  upload_face(idx: number, file: File, name: string): void {
    this.progressInfos[idx] = { value: 0, fileName: name };

    if (file) {
      this.uploadService.upload_face(file).subscribe({
        next: (event: any) => {
          if (event.type === HttpEventType.UploadProgress) {
            this.progressInfos[idx].value = Math.round(100 * event.loaded / event.total);
          } else if (event instanceof HttpResponse) {
            const msg = 'Uploaded the file successfully: ' + file.name;
            this.message.push(msg);
            this.imageInfos = this.uploadService.getFiles();
          }
        },
        error: (err: any) => {
          this.progressInfos[idx].value = 0;

          const msg = 'Could not upload the file: ' + file.name;
          this.message.push(msg);
        }});
    }
  }

  uploadFiles(): void {
    console.log("Click hair style");

    this.message = [];

    if (this.selectedFiles1) {
      console.log('Upload image');
      // for (let i = 0; i < this.selectedFiles1.length; i++) {
      this.upload_face(0, this.selectedFiles1[0], 'src');
      // }
    }

  }

  uploadFiles2(): void {
    console.log("Click hair style");

    this.message = [];

    if (this.selectedFiles1) {
      console.log('Upload image');
      // for (let i = 0; i < this.selectedFiles1.length; i++) {
      this.upload_face2(0, this.selectedFiles1[0], 'src');
      // }
    }

  }

}
