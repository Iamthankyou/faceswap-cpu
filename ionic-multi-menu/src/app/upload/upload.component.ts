import { Component, OnInit } from '@angular/core';
import {Observable} from "rxjs";
import {FileUploadService} from "../file-upload.service";
import {HttpEventType, HttpResponse} from "@angular/common/http";

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',
  styleUrls: ['./upload.component.scss'],
})
export class UploadComponent implements OnInit {
  selectedFiles1?: FileList;
  selectedFiles2?: FileList;

  progressInfos: any[] = [];
  message: string[] = [];

  previews1: string[] = [];
  previews2: string[] = [];

  previews3: string[] = []
  imageInfos?: Observable<any>;

  constructor(private uploadService: FileUploadService) { }

  ngOnInit(): void {
    this.imageInfos = this.uploadService.getFiles();
    this.previews1.push(`http://localhost:5000/src?${Date.now()}`);
    this.previews2.push(`http://localhost:5000/dst?${Date.now()}`);
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
    console.log("Swapface");
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

  upload_src(idx: number, file: File, name: string): void {
    this.progressInfos[idx] = { value: 0, fileName: name };

    if (file) {
      this.uploadService.upload_src(file).subscribe({
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

  upload_dst(idx: number, file: File, name: string): void {
    this.progressInfos[idx] = { value: 0, fileName: name };

    if (file) {
    this.uploadService.upload_dst(file).subscribe({
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
        },
    });
    }
  }

  uploadFiles(): void {
    this.message = [];

    if (this.selectedFiles1) {
        this.upload_src(0, this.selectedFiles1[0],'src');
    }

    if (this.selectedFiles2) {
        this.upload_dst(0, this.selectedFiles2[0],'dst');
    }
  }

}
