import { FormEvent, useEffect, useState } from "react";
import { createProduct, listProducts } from "../api/products";
import PageHeader from "../components/PageHeader";
import type { ProductCreateRequest, ProductResponse } from "../types/product";

const initialForm: ProductCreateRequest = {
  product_code: "",
  brand: "",
  name: "",
  model: "",
  category: "",
  description: "",
  price: 0,
  currency: "CNY",
  stock_quantity: 0,
  sale_status: "on_sale",
  features: [],
  use_cases: [],
  specifications: {},
  tags: [],
  popularity_score: 0,
  is_active: true
};

function splitValues(value: string): string[] {
  return value
    .split(/[,，]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export default function ProductsPage() {
  const [products, setProducts] = useState<ProductResponse[]>([]);
  const [form, setForm] = useState<ProductCreateRequest>(initialForm);
  const [features, setFeatures] = useState("");
  const [useCases, setUseCases] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  useEffect(() => {
    void refreshProducts();
  }, []);

  async function refreshProducts() {
    setLoading(true);
    setError("");
    try {
      setProducts((await listProducts()).items);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : String(requestError));
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setMessage("");
    try {
      const created = await createProduct({
        ...form,
        features: splitValues(features),
        use_cases: splitValues(useCases)
      });
      setProducts((current) => [created, ...current]);
      setForm(initialForm);
      setFeatures("");
      setUseCases("");
      setMessage(`商品 ${created.product_code} 已创建`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : String(requestError));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-stack">
      <PageHeader
        eyebrow="Products"
        title="商品录入"
        description="创建智能客服可查询和推荐的商品。"
      />
      <div className="products-layout">
        <form className="card product-form" onSubmit={handleSubmit}>
          <h3>新增商品</h3>
          <div className="product-form-row">
            <label>商品编码<input required value={form.product_code} onChange={(event) => setForm({ ...form, product_code: event.target.value })} /></label>
            <label>品牌<input required value={form.brand} onChange={(event) => setForm({ ...form, brand: event.target.value })} /></label>
          </div>
          <label>商品名称<input required value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} /></label>
          <div className="product-form-row">
            <label>型号<input required value={form.model} onChange={(event) => setForm({ ...form, model: event.target.value })} /></label>
            <label>分类<input required value={form.category} onChange={(event) => setForm({ ...form, category: event.target.value })} /></label>
          </div>
          <div className="product-form-row">
            <label>价格<input required min="0" step="0.01" type="number" value={form.price} onChange={(event) => setForm({ ...form, price: Number(event.target.value) })} /></label>
            <label>库存<input required min="0" type="number" value={form.stock_quantity} onChange={(event) => setForm({ ...form, stock_quantity: Number(event.target.value) })} /></label>
          </div>
          <label>商品特征（逗号分隔）<input value={features} onChange={(event) => setFeatures(event.target.value)} placeholder="容易清洗，低噪音" /></label>
          <label>使用场景（逗号分隔）<input value={useCases} onChange={(event) => setUseCases(event.target.value)} placeholder="两人家庭，宿舍" /></label>
          <label>商品描述<textarea value={form.description ?? ""} onChange={(event) => setForm({ ...form, description: event.target.value })} /></label>
          {error && <p className="form-error">{error}</p>}
          {message && <p className="form-success">{message}</p>}
          <button type="submit" disabled={loading}>{loading ? "提交中..." : "创建商品"}</button>
        </form>

        <section className="card">
          <div className="panel-header">
            <h3>商品列表</h3>
            <button className="secondary-button" type="button" onClick={() => void refreshProducts()} disabled={loading}>刷新</button>
          </div>
          {products.length === 0 ? (
            <p>{loading ? "加载中..." : "暂无商品"}</p>
          ) : (
            <div className="product-list">
              {products.map((product) => (
                <article className="product-item" key={product.id}>
                  <div className="panel-header">
                    <strong>{product.name}</strong>
                    <span className="product-status">{product.sale_status}</span>
                  </div>
                  <p>{product.brand} · {product.model} · {product.category}</p>
                  <small>{product.product_code} · ¥{product.price} · 库存 {product.stock_quantity}</small>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
