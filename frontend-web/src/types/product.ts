export type ProductCreateRequest = {
  product_code: string;
  brand: string;
  name: string;
  model: string;
  category: string;
  description?: string | null;
  price: number;
  currency: string;
  stock_quantity: number;
  sale_status: string;
  features: string[];
  use_cases: string[];
  specifications: Record<string, unknown>;
  tags: string[];
  popularity_score: number;
  is_active: boolean;
};

export type ProductResponse = ProductCreateRequest & {
  id: number;
  official_product_url?: string | null;
  source_checked_at?: string | null;
};

export type ProductListResponse = {
  items: ProductResponse[];
  total: number;
  page: number;
  page_size: number;
};

export type ProductDocumentLink = {
  id: number;
  product_id: number;
  document_id: number;
  document_type: string;
  is_primary: boolean;
  manual_version?: string | null;
};
