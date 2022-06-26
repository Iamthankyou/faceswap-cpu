import { Injectable } from '@angular/core';
import {Observable} from "rxjs";
import {HttpClient, HttpEvent, HttpRequest} from "@angular/common/http";

@Injectable({
  providedIn: 'root'
})
export class FileUploadService {
  private baseUrl = 'http://0.0.0.0:5000';

  constructor(private http: HttpClient) { }

  upload_src(file: File): Observable<HttpEvent<any>> {
    const formData: FormData = new FormData();

    formData.append('file', file);

    const req = new HttpRequest('POST', `${this.baseUrl}/upload_src`, formData, {
      reportProgress: true,
      responseType: 'json'
    });

    return this.http.request(req);
  }

  upload_dst(file: File): Observable<HttpEvent<any>> {
    const formData: FormData = new FormData();

    formData.append('file', file);

    const req = new HttpRequest('POST', `${this.baseUrl}/upload_dst`, formData, {
      reportProgress: true,
      responseType: 'json'
    });

    return this.http.request(req);
  }

  upload_face(file: File): Observable<HttpEvent<any>> {
    const formData: FormData = new FormData();

    formData.append('file', file);

    const req = new HttpRequest('POST', `${this.baseUrl}/hair_style2`, formData, {
      reportProgress: true,
      responseType: 'json'
    });

    return this.http.request(req);
  }

  upload_face2(file: File): Observable<HttpEvent<any>> {
    const formData: FormData = new FormData();

    formData.append('file', file);

    const req = new HttpRequest('POST', `${this.baseUrl}/hair_style3`, formData, {
      reportProgress: true,
      responseType: 'json'
    });

    return this.http.request(req);
  }

  upload_video(file: File): Observable<HttpEvent<any>> {
    const formData: FormData = new FormData();

    formData.append('file', file);

    const req = new HttpRequest('POST', `${this.baseUrl}/upload_video`, formData, {
      reportProgress: true,
      responseType: 'json'
    });

    return this.http.request(req);
  }

  getVideo():Observable<any> {
    console.log('get video');
    return this.http.get(`${this.baseUrl}/get_video`);
  }

  getFiles(): Observable<any> {
      console.log('Get files');
      return this.http.get(`${this.baseUrl}/files`, {responseType : 'blob'});
  }

  getNumber(): Observable<any> {
    console.log('Get number');
    return this.http.get(`${this.baseUrl}/get_number`);
  }

  swapStyle(name:any): Observable<any> {
    console.log('Get number');
    return this.http.get(`${this.baseUrl}/swap_style?name=`+name);
  }


  swapface(): Observable<any> {
    return this.http.get(`${this.baseUrl}/swap_images`);
  }
}
