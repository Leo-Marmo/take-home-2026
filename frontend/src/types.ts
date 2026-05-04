/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export interface Category {
  name: string;
}
export interface Price {
  price: number;
  currency: string;
  compare_at_price?: number | null;
}
export interface Product {
  name: string;
  price: Price;
  description: string;
  key_features: string[];
  image_urls: string[];
  video_url?: string | null;
  category: Category;
  brand: string;
  colors: string[];
  variants: Variant[];
}
export interface Variant {
  options: VariantOption[];
  sku?: string | null;
  price?: Price | null;
  image_url?: string | null;
  in_stock?: boolean;
}
export interface VariantOption {
  name: string;
  value: string;
}
export interface ProductSummary {
  id: string;
  product: Product;
}
