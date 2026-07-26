import type {
  ProductCreateRequest,
  ProductListResponse,
  ProductResponse
} from "../types/product";
import { apiRequest } from "./client";

export function listProducts() {
  return apiRequest<ProductListResponse>("/products?page=1&page_size=100");
}

export function createProduct(data: ProductCreateRequest) {
  return apiRequest<ProductResponse>("/products", {
    method: "POST",
    body: data
  });
}
